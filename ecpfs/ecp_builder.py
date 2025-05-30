import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import math
import time
import zarr
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from collections import deque

from ._logging import logger
from .utils import (
    Metric,
    determine_node_assignments,
    get_source_embeddings,
    calculate_chunk_size,
)


class ECPBuilder:
    """
    Class to build the eCP index tree.
    The class is designed to build a hierarchical index for
    fast nearest neighbor search in high-dimensional spaces.
    """

    enable_time_logging = True

    def log_time(func):
        """
        Decorator to log the execution time of a function.
        """

        def wrapper(*args, **kwargs):
            if not ECPBuilder.enable_time_logging:
                return func(*args, **kwargs)
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            args[0].logger.info(
                f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds."
            )
            return result

        return wrapper

    def __init__(
        self,
        levels: int,
        target_cluster_items=100,
        metric=Metric.L2,
        index_file="ecpfs_index.zarr",
        file_store="zarr_l",
        workers=4,
        memory_limit=4
    ):
        """
        Constructor.

        Parameters:
            levels: Number of levels in the index hierarchy
            target_cluster_items: Aim for clusters with this many items (no guarantees)
            metric: Metric to use when building the index [L2 (Euclidean) | IP (Inner Product) | cos (Cosine Similarity)]
            index_file: If a filename is specified, all index related data will be stored in a store. Default behaviour is to store it using a zarr.storage.LocalStore under the name "ecpfs_index.zarr". Set this to empty string if you do not want to store the index.
            file_store: The file format to store the representative cluster embeddings and ids. "zarr_l" (LocalStore) or "zarr_z" (ZipStore), default="zarr_l".
            workers: Number of worker threads to use for parallel processing
            memory_threshold: Determines when to use multithreading when inserting data into the index. Default is 200 MB.
            memory_limit: The amount of memory available for the in GB. Default is 4 GB.
        """
        self.levels: int = levels
        self.target_cluster_items: int = target_cluster_items
        self.logger = logger
        self.metric: Metric = metric
        self.index_file = index_file
        self.file_store = file_store
        self.representative_ids: np.ndarray | None = None
        self.representative_embeddings: np.ndarray | None = None
        self.chunk_size = (-1, -1)
        self.workers = cpu_count() - 1 if workers > cpu_count() else workers
        self.memory_limit = memory_limit * 1024 * 1024 * 1024

        self.write_index_info()
        return

    @log_time
    def select_cluster_representatives(
        self,
        embeddings_file: Path,
        option="offset",
        grp_name="embeddings",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the list of representatives for clusters.

        Parameters:
            embeddings_file: A .h5/.zarr/.zip/.zipstore file that stores 2D np.ndarray of embeddings (.zarr = LocalStore, .zip/.zipstore = ZipStore)

            option: "offset", "random", or "dissimilar".
                "offset" = select based of an offset determined through the index config info,
                "random" = arbitrarily select cluster representatives, and
                "dissimilar" = ensure that a representative is dissimilar from others at the same level.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays:
                - embeddings: 2D array of representative cluster embeddings.
                - ids: 1D array of representative cluster ids.
        """
        logger.info(f"Selecting cluster representatives using {option}...")
        embeddings = get_source_embeddings(embeddings_file, grp_name)
        self.itemsize = embeddings.dtype.itemsize
        self.emb_dtype = embeddings.dtype
        self.emb_dims = embeddings.shape[1]

        # Determine sizes
        (total_items, total_features) = embeddings.shape
        self.total_clusters = math.ceil(total_items / self.target_cluster_items)
        self.node_size = math.ceil(self.total_clusters ** (1.0 / self.levels))

        self.chunk_size = calculate_chunk_size(
            num_features=total_features, dtype=embeddings.dtype
        )

        self.logger.info(f"Chunk Shape = {self.chunk_size}")

        # Select cluster centroids
        if option == "offset":
            all_item_ids = np.arange(total_items).astype(np.uint32)
            self.representative_ids = all_item_ids[:: self.target_cluster_items]
            self.logger.info(
                f"{len(self.representative_ids)} Cluster representatives selected"
            )
            if len(self.representative_ids) != self.total_clusters:
                self.logger.error(
                    "Number of representatives does not match the total clusters."
                )
                raise ValueError(
                    f"Number of representatives does not match the total clusters. {total_items, len(self.representative_ids), self.total_clusters}"
                )
            self.logger.info(f"Getting representative embeddings from file")
            self.representative_embeddings = embeddings[:: self.target_cluster_items]
        elif option == "random":
            if self.total_clusters > total_items:
                self.logger.error(
                    "Total clusters exceed the number of available embeddings."
                )
            self.representative_ids = np.random.choice(
                total_items, size=self.total_clusters, replace=False
            )
            self.logger.info(
                f"{len(self.representative_ids)} Cluster Representatives Selected"
            )
            self.logger.info(f"Getting representative embeddings from file")
            self.representative_embeddings = embeddings[self.representative_ids]
        elif option == "mbk":
            mbk = MiniBatchKMeans(
                init="k-means++", n_clusters=self.total_clusters, batch_size=50000
            )
            for start in tqdm(range(0, embeddings.shape[0], 200000)):
                end = min(start + 200000, embeddings.shape[0])
                mbk.partial_fit(embeddings[start:end])
            self.representative_embeddings = mbk.cluster_centers_
            self.representative_ids = np.arange(self.total_clusters)
        elif option == "dissimilar":
            if self.metric == Metric.IP:
                """"""
            elif self.metric == Metric.L2:
                """"""
            elif self.metric == Metric.COS:
                """"""
            raise NotImplementedError("dissimilar option")
        else:
            raise ValueError(
                "Unknown option, the valid options are ['offset', 'random', 'mbk', 'dissimilar']"
            )

        self.write_cluster_representatives()

        return self.representative_embeddings, self.representative_ids

    @log_time
    def get_cluster_representatives_from_file(
        self,
        fp: Path,
        emb_dsname="rep_embeddings",
        ids_dsname="rep_item_ids",
    ) -> Tuple[zarr.Array, zarr.Array]:
        """
        Load cluster representatives from a zarr (local or zip) store.

        Parameters:
            fp (Path): File path to the zarr or HDF5 file.
            emb_dsname (str): Dataset name of representative embeddings, default=rep_embeddings
            ids_dsname (str): Dataset name of representative item ids, default=rep_item_ids

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays:
                - embeddings: 2D array of representative cluster embeddings.
                - ids: 1D array of representative cluster ids.
        """
        logger.info("Loading cluster representatives...")

        zf = zarr.open(fp, mode="r")
        self.representative_embeddings = zf[emb_dsname]
        self.representative_ids = zf[ids_dsname]
        self.representative_emb_dsname = emb_dsname
        self.representative_ids_dsname = ids_dsname

        self.total_clusters = self.representative_ids.shape[0]
        self.node_size = math.ceil(self.total_clusters ** (1.0 / self.levels))

        self.chunk_size = calculate_chunk_size(
            num_features=self.representative_embeddings.shape[1],
            dtype=self.representative_embeddings.dtype,
        )
        self.logger.info(f"Chunk Shape = {self.chunk_size}")

        self.write_cluster_representatives()

        return self.representative_embeddings, self.representative_ids

    @log_time
    def write_index_info(self):
        root = None
        if self.file_store == "zarr_l":
            root = zarr.open(self.index_file, mode="w")
        elif self.file_store == "zarr_z":
            root = zarr.open(zarr.storage.ZipStore(self.index_file, mode="w"), mode="w")
        root.create_group("info")
        root["info"]["levels"] = np.array([self.levels])
        root["info"]["metric"] = np.array([self.metric.value])

    @log_time
    def write_cluster_representatives(self):
        self.rep_emb_dsname = "rep_embeddings"
        self.rep_ids_dsname = "rep_item_ids"
        root = None
        if self.file_store == "zarr_l":
            root = zarr.open(self.index_file)
        elif self.file_store == "zarr_z":
            root = zarr.open(zarr.storage.ZipStore(self.index_file, mode="a"), mode="a")

        root[self.rep_ids_dsname] = self.representative_ids
        root.create_array(
            name=self.rep_emb_dsname,
            shape=self.representative_embeddings.shape,
            dtype=self.representative_embeddings.dtype,
            chunks=self.chunk_size,
        )
        root[self.rep_emb_dsname] = self.representative_embeddings

    def update_node_border_info(self, lvl: str, node: str):
        """
        Calculate and set the border item for the node at the provided level

        Parameters:
            lvl (str): The level of the node, ex. "lvl_0", "lvl_1"
            node (str): The node to update, ex. "node_0", "node_1"
        """
        # Update border info
        if self.metric == Metric.IP or self.metric == Metric.COS:
            new_border_dists = np.argsort(np.sort(self.index[lvl][node]["distances"]))
            self.index[lvl][node]["border"] = (
                new_border_dists[0],
                self.index[lvl][node]["distances"][new_border_dists[0]],
            )
        elif self.metric == Metric.L2:
            new_border_dists = np.argsort(np.sort(self.index[lvl][node]["distances"]))[
                ::-1
            ]
            self.index[lvl][node]["border"] = (
                new_border_dists[0],
                self.index[lvl][node]["distances"][new_border_dists[0]],
            )

    def add_data_to_index_level(
        self,
        embeddings: np.ndarray,
        tasks: list,
        target_lvl,
        proc_workers=1,
    ):
        """
        Add data to the desired target index level.
        Does it in a DFS manner.

        Parameters:
            data_indices: The indices of the data to add.
            max_lvl: The maximum level to add the data to.
        """
        buckets = {1: deque(tasks)}
        while buckets:
            # Find the single deepest level that has any tasks
            current_lvl = max(buckets.keys())
            queue = buckets[current_lvl]

            # Pop exactly ONE task from that level
            lvl, node_idx, d_idxs = queue.popleft()
            # If that deque is now empty, remove it
            if not queue:
                del buckets[current_lvl]

            lvl_s = f"lvl_{lvl}"
            node_s = f"node_{node_idx}"
            # If target_lvl is reached write the node and continue
            if lvl == target_lvl:
                ids_key = "node_ids" if target_lvl != self.levels else "item_ids"
                if "embeddings" in self.index[lvl_s][node_s]:
                    self.index[lvl_s][node_s]["embeddings"].append(embeddings[d_idxs])
                    self.index[lvl_s][node_s][ids_key].append(
                        self.batch_to_global_ids[d_idxs]
                    )
                    continue
                zarr.create_array(
                    f"{self.index_file}/{lvl_s}/{node_s}",
                    name="embeddings",
                    data=embeddings[d_idxs],
                    chunks=self.chunk_size,
                )
                zarr.create_array(
                    f"{self.index_file}/{lvl_s}/{node_s}",
                    name=ids_key,
                    data=self.batch_to_global_ids[d_idxs],
                    chunks=(self.chunk_size[0]),
                )
                zarr.create_array(
                    f"{self.index_file}/{lvl_s}/{node_s}",
                    name="border",
                    shape=(2,),
                    dtype=np.float32,
                )
                continue

            # If target_lvl has not been reached determine the next level assignments
            centroids = self.index[lvl_s][node_s]["embeddings"][:]

            offsets, data = determine_node_assignments(
                node_embeddings=centroids,
                data_embeddings=embeddings[d_idxs],
                workers=proc_workers,
                metric=self.metric
            )

            node_ids = self.index[lvl_s][node_s]["node_ids"][:]
            # Update next level queue in buckets
            nxt = buckets.setdefault(lvl + 1, deque())
            for child in range(offsets.shape[0] - 1):
                start, end = offsets[child], offsets[child + 1]
                if start == end:
                    continue
                # data eleemnts match local ids
                # d_idxs have the batch ids
                child_data = d_idxs[data[start:end]]
                nxt.append((lvl + 1, node_ids[child], child_data))

    @log_time
    def build_tree_fs(
        self,
        embeddings_file: Path,
        grp_name="embeddings",
    ) -> None:
        """
        Build the hierarchical eCP index tree for an and store it into a file.
        """

        self.logger.info("Building tree...")

        self.index = None
        if self.file_store == "zarr_l":
            self.index = zarr.open(self.index_file)
        elif self.file_store == "zarr_z":
            self.index = zarr.open(
                zarr.storage.ZipStore(self.index_file, mode="a"), mode="a"
            )

        self.root_embeddings = self.representative_embeddings[: self.node_size]

        # /index_root
        self.index.create_group("index_root")
        # /index_root/embeddings (N,D) float16/float32
        zarr.create_array(
            f"{self.index_file}/index_root/embeddings",
            data=self.root_embeddings,
            chunks=self.chunk_size,
        )
        # /index_root/node_ids (N,) uint32/uint64
        zarr.create_array(
            f"{self.index_file}/index_root/node_ids",
            data=self.representative_ids[: self.node_size],
        )

        self.logger.info("Constructing levels...")
        lvl_range = 1
        for l in range(1, self.levels + 1):
            lvl_range = lvl_range * self.node_size
            if lvl_range > self.total_clusters:
                lvl_range = self.total_clusters
            lvl = f"lvl_{l}"
            self.index.create_group(lvl)
            for i in range(lvl_range):
                node = f"node_{i}"
                self.index[lvl].create_group(node)

        # Start building tree top-down, root -> lvl_0 -> lvl_1 -> ... -> lvl_n-1
        self.logger.info("Adding representatives top-down...")
        lvl_range = self.node_size

        T = self.workers
        batch_bytes = self.memory_limit // 2
        batch_size = batch_bytes // (self.emb_dims * self.itemsize)
        for l in range(1, self.levels + 1):
            if l == self.levels:
                self.logger.info("Adding dataset items...")
                embeddings = get_source_embeddings(embeddings_file, grp_name)
                lvl_range = embeddings.shape[0]
            else:
                embeddings = self.representative_embeddings
                lvl_range = min(lvl_range * self.node_size, self.total_clusters)

            for start in range(0, lvl_range, batch_size):
                end = min(start + batch_size, lvl_range)
                batch_embeddings = embeddings[start:end]
                # Set batch id to global id array
                self.batch_to_global_ids = np.arange(start, end, dtype=np.uint32)

                offsets, data = determine_node_assignments(
                    node_embeddings=self.root_embeddings,
                    data_embeddings=batch_embeddings,
                    workers=T,
                    metric=self.metric
                )
                self.logger.info(f"Root done")

                tasks = []
                for i in range(offsets.shape[0] - 1):
                    start, end = offsets[i], offsets[i + 1]
                    if start == end:
                        continue
                    # data elements match batch ids
                    ids = data[start:end]
                    tasks.append((1, i, ids))

                def process_node(task):
                    # this will only go as deep as target_lvl=l
                    self.add_data_to_index_level(
                        embeddings=batch_embeddings,
                        tasks=[task],
                        target_lvl=l,
                        proc_workers=1,
                    )

                futures = []
                with ThreadPoolExecutor(max_workers=T) as pool:
                    for task in tasks:
                        futures.append(pool.submit(process_node, task))

                    for fut in as_completed(futures):
                        try:
                            _ = fut.result()
                        except Exception as e:
                            self.logger.error(
                                f"process_node failed for task {task}: {e}"
                            )
                            raise

        self.logger.info("Done building index!")
