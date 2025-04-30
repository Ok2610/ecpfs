import concurrent.futures
import math
import concurrent
import time
import h5py
import zarr
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from multiprocessing import cpu_count

from ecpfs.utils import (
    Metric,
    calculate_distances,
    get_source_embeddings,
    calculate_chunk_size,
)


class ECPBuilder:
    """
    Class to build the eCP index tree.
    The class is designed to build a hierarchical index for fast nearest neighbor search in high-dimensional spaces.
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
        logger,
        target_cluster_items=100,
        metric=Metric.L2,
        index_file="ecpfs_index.zarr",
        file_store="zarr_l",
        workers=4,
    ):
        """
        Constructor.

        Parameters:

            levels: Number of levels in the index hierarchy
            target_cluster_items: Aim for clusters with this many items (no guarantees)
            metric: Metric to use when building the index [L2 (Euclidean) | IP (Inner Product) | cos (Cosine Similarity)]
            index_file: If a filename is specified, all index related data will be stored in a store. Default behaviour is to store it using a zarr.storage.LocalStore under the name "ecpfs_index.zarr". Set this to empty string if you do not want to store the index.
            file_store: The file format to store the representative cluster embeddings and ids. "zarr_l" (LocalStore), "zarr_z" (ZipStore), or "h5" (HDF5), default="zarr_l".
        """
        self.levels: int = levels
        self.target_cluster_items: int = target_cluster_items
        self.logger = logger
        self.metric: Metric = metric
        self.index_file = index_file
        self.file_store = file_store
        self.representative_ids: np.ndarray | None = None
        self.representative_embeddings: np.ndarray | None = None
        # self.item_to_cluster_map = {}
        self.chunk_size = (-1, -1)
        self.workers = workers

        self.write_index_info()
        return

    @log_time
    def select_cluster_representatives(
        self,
        embeddings_file: Path,
        option="offset",
        grp=False,
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
        embeddings = get_source_embeddings(embeddings_file, grp, grp_name)

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
        # Store the cluster representative embeddings and item ids

        return self.representative_embeddings, self.representative_ids

    @log_time
    def get_cluster_representatives_from_file(
        self,
        fp: Path,
        emb_dsname="rep_embeddings",
        ids_dsname="rep_item_ids",
        format="zarr_l",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load cluster representatives from a zarr (local or zip) store or a HDF5 file.

        The HDF5 file should have two datasets, one for their representative embeddings and one for their item ids.

        Parameters:
            fp (Path): File path to the zarr or HDF5 file.
            emb_dsname (str): Dataset name of representative embeddings, default=rep_embeddings
            ids_dsname (str): Dataset name of representative item ids, default=rep_item_ids
            format (str): Format of the file. "zarr_l" (LocalStore), "zarr_z" (ZipStore), or "h5" (HDF5), default="zarr_l".

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays:
                - embeddings: 2D array of representative cluster embeddings.
                - ids: 1D array of representative cluster ids.
        """

        if format == "zarr_l":
            zf = zarr.open(fp, mode="r")
            self.representative_embeddings = zf[emb_dsname][:]
            self.representative_ids = zf[ids_dsname][:]
        elif format == "zarr_z":
            with zarr.storage.ZipStore(fp, mode="r") as store:
                zf = zarr.open(store, mode="r")
                self.representative_embeddings = zf[emb_dsname][:]
                self.representative_ids = zf[ids_dsname][:]
        elif format == "h5":
            with h5py.File(fp, "r") as hf:
                self.representative_embeddings = hf[emb_dsname][:]
                self.representative_ids = hf[ids_dsname][:]

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
        if self.file_store == "zarr_l" or self.file_store == "zarr_z":
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file, mode="w")
            elif self.file_store == "zarr_z":
                root = zarr.open(
                    zarr.storage.ZipStore(self.index_file, mode="w"), mode="w"
                )
            root.create_group("info")
            root["info"]["levels"] = np.array([self.levels])
            root["info"]["metric"] = np.array([self.metric.value])
        elif self.file_store == "h5":
            with h5py.File(self.index_file, "a") as hf:
                hf.create_group("info")
                hf["info"]["levels"] = self.levels
                hf["info"]["metric"] = self.metric.value

    @log_time
    def write_cluster_representatives(self):
        rep_ids_dsname = "rep_item_ids"
        rep_emb_dsname = "rep_embeddings"
        if self.file_store == "zarr_l" or self.file_store == "zarr_z":
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file)
            elif self.file_store == "zarr_z":
                root = zarr.open(
                    zarr.storage.ZipStore(self.index_file, mode="a"), mode="a"
                )

            root[rep_ids_dsname] = self.representative_ids
            root.create_array(
                name=rep_emb_dsname,
                shape=self.representative_embeddings.shape,
                dtype=self.representative_embeddings.dtype,
                chunks=self.chunk_size,
            )
            root[rep_emb_dsname] = self.representative_embeddings
        elif self.file_store == "h5":
            with h5py.File(self.index_file, "a") as hf:
                hf[rep_ids_dsname] = self.representative_ids
                hf[rep_emb_dsname] = self.representative_embeddings

    def align_specific_node(self, lvl, node):
        """
        Resize the embeddings and distances arrays of a node to match their actual size

        Parameters:
            lvl (str): The level of the node, ex. "lvl_0", "lvl_1"
            node (str): The node to resize, ex. "node_0", "node_1"
        """
        self.index[lvl][node]["embeddings"].resize(
            (
                len(self.index[lvl][node]["item_ids"]),
                self.index[lvl][node]["embeddings"].shape[1],
            )
        )
        self.index[lvl][node]["distances"].resize(
            len(self.index[lvl][node]["item_ids"])
        )
        if len(self.index[lvl][node]["item_ids"]) > 0:
            self.update_node_border_info(lvl, node)

    def update_node_border_info(self, lvl: str, node: str):
        """
        Calculate and set the border item for the node at the provided level

        Parameters:
            lvl (str): The level of the node, ex. "lvl_0", "lvl_1"
            node (str): The node to update, ex. "node_0", "node_1"
        """
        # Update border info
        if self.metric == Metric.IP or self.metric == Metric.COS:
            # TODO: Calculate dot product with representative embedding
            new_border_dists = np.argsort(np.sort(self.index[lvl][node]["distances"]))
            self.index[lvl][node]["border"] = (
                new_border_dists[0],
                self.index[lvl][node]["distances"][new_border_dists[0]],
            )
        elif self.metric == Metric.L2:
            # TODO: Calculate L2 distance with representative embedding
            new_border_dists = np.argsort(np.sort(self.index[lvl][node]["distances"]))[
                ::-1
            ]
            self.index[lvl][node]["border"] = (
                new_border_dists[0],
                self.index[lvl][node]["distances"][new_border_dists[0]],
            )

    def align_node_embeddings_and_distances(self):
        """
        Resize the embeddings and distances arrays of all nodes to match their actual size
        """
        lvl_range = self.node_size
        for i in range(self.levels):
            lvl = "lvl_" + str(i)
            level = int(lvl.split("_")[1])
            if level == self.levels - 1:
                ids_key = "item_ids"
            else:
                ids_key = "node_ids"

            for j in range(lvl_range):
                node = "node_" + str(j)
                self.index[lvl][node]["embeddings"].resize(
                    (
                        len(self.index[lvl][node][ids_key]),
                        self.index[lvl][node]["embeddings"].shape[1],
                    )
                )
                self.index[lvl][node]["distances"].resize(
                    len(self.index[lvl][node][ids_key])
                )
                if len(self.index[lvl][node][ids_key]) > 0:
                    self.update_node_border_info(lvl, node)
            if i + 1 == self.levels - 1:
                lvl_range = self.total_clusters
            else:
                lvl_range = lvl_range * self.node_size

    @log_time
    def build_tree_fs(self) -> None:
        """
        Build the hierarchical eCP index tree for an and store it into a file.
        """

        def check_to_replace_border(
            lvl: str, node: str, emb: np.ndarray, cl_idx: int, dist: float
        ):
            """
            Check if the maximum number of items have been reached in a node, and
            if the new item is better than the border item, replace and update border item info

            Parameters:
                lvl (str): The level of the node, ex. "lvl_0", "lvl_1"
                node (str): The node to check, ex. "node_0", "node_1"
                emb (np.ndarray): The embedding vector of the item to check
                cl_idx (int): The index of the representative cluster
                dist (float): The distance of the item to the node
            """
            border_idx, border_dist = self.index[lvl][node]["border"]
            if self.metric == Metric.IP or self.metric == Metric.COS:
                if border_dist < dist:
                    self.index[lvl][node]["embeddings"][border_idx] = emb
                    self.index[lvl][node]["items_ids"][border_idx] = (
                        self.representative_ids[cl_idx]
                    )
                    self.index[lvl][node]["node_ids"][border_idx] = cl_idx
                    self.index[lvl][node]["distances"][border_idx] = dist
                    self.update_node_border_info(lvl, node)
            elif self.metric == Metric.L2:
                if border_dist > dist:
                    self.index[lvl][node]["embeddings"][border_idx] = emb
                    self.index[lvl][node]["items_ids"][border_idx] = (
                        self.representative_ids[cl_idx]
                    )
                    self.index[lvl][node]["node_ids"][border_idx] = cl_idx
                    self.index[lvl][node]["distances"][border_idx] = dist
                    self.update_node_border_info(lvl, node)

        if self.representative_embeddings is None or self.representative_ids is None:
            raise ValueError(
                """
                Cluster representatives not determined! 
                Set recompute to True, or call select_cluster_representatives(...) prior to calling build_tree(...)
                """
            )

        self.index = {}
        self.index["root"] = {}
        self.index["root"]["node_ids"] = self.representative_ids[: self.node_size]
        self.index["root"]["embeddings"] = self.representative_embeddings[
            : self.node_size
        ]

        self.logger.info("Constructing levels...")
        lvl_range = 1
        for l in range(self.levels):
            lvl_range = lvl_range * self.node_size
            if lvl_range > self.total_clusters:
                lvl_range = self.total_clusters
            lvl = "lvl_" + str(l)
            if l == self.levels - 1:
                ids_key = "item_ids"
            else:
                ids_key = "node_ids"
            self.index[lvl] = {}
            for i in range(lvl_range):
                node = "node_" + str(i)
                self.index[lvl][node] = {
                    "embeddings": np.zeros(
                        shape=(self.node_size, self.representative_embeddings.shape[1]),
                        dtype=self.representative_embeddings.dtype,
                    ),
                    ids_key: [],
                    "distances": np.zeros(
                        shape=(self.node_size,),
                        dtype=self.representative_embeddings.dtype,
                    ),
                    "border": (-1, 0.0),
                }

        # Start building tree top-down
        # As we already have root node, we start by inserting the nodes of lvl_1 into lvl_0.
        # Then we input lvl_2 items into lvl_1, until we reach self.levels-1.
        self.logger.info("Adding representatives top-down...")
        lvl_range = self.node_size
        for l in range(self.levels - 1):
            lvl_range = lvl_range * self.node_size
            if lvl_range > self.total_clusters:
                lvl_range = self.total_clusters
            embeddings = self.representative_embeddings[:lvl_range]
            for cl_idx, emb in enumerate(embeddings):
                top, distances = calculate_distances(
                    emb, self.index["root"]["embeddings"], self.metric
                )
                n = top[0]
                curr_lvl = 0
                while True:
                    lvl = "lvl_" + str(curr_lvl)
                    node = "node_" + str(n)
                    if curr_lvl == l:
                        next = len(self.index[lvl][node]["node_ids"])
                        if next >= self.index[lvl][node]["embeddings"].shape[0]:
                            concat_array_emb = np.zeros(
                                shape=(
                                    self.node_size,
                                    self.representative_embeddings.shape[1],
                                ),
                                dtype=self.representative_embeddings.dtype,
                            )
                            concat_array_dist = np.zeros(
                                shape=(self.node_size,),
                                dtype=self.representative_embeddings.dtype,
                            )
                            self.index[lvl][node]["embeddings"] = np.concatenate(
                                (self.index[lvl][node]["embeddings"], concat_array_emb)
                            )
                            self.index[lvl][node]["distances"] = np.concatenate(
                                (self.index[lvl][node]["distances"], concat_array_dist)
                            )
                        self.index[lvl][node]["embeddings"][next] = emb
                        self.index[lvl][node]["node_ids"].append(cl_idx)
                        self.index[lvl][node]["distances"][next] = distances[top[0]]
                        # TODO: Move the below part to a function and call it after index is built
                        if self.metric == Metric.IP or self.metric == Metric.COS:
                            if (
                                self.index[lvl][node]["border"][1] is None
                                or self.index[lvl][node]["border"][1]
                                > distances[top[0]]
                            ):
                                self.index[lvl][node]["border"] = (
                                    next,
                                    distances[top[0]],
                                )
                        elif self.metric == Metric.L2:
                            if (
                                self.index[lvl][node]["border"][1] is None
                                or self.index[lvl][node]["border"][1]
                                < distances[top[0]]
                            ):
                                self.index[lvl][node]["border"] = (
                                    next,
                                    distances[top[0]],
                                )
                        break
                    else:
                        top, distances = calculate_distances(
                            emb, self.index[lvl][node]["embeddings"], self.metric
                        )
                        n = self.index[lvl][node]["node_ids"][top[0]]
                    curr_lvl += 1

        self.logger.info("Aligning arrays...")
        # Resize the embeddings and distance arrays of nodes to actual size
        self.align_node_embeddings_and_distances()
        self.logger.info("Saving tree to file...")
        self.write_tree_structure_to_file()
        self.logger.info("Done building tree!")

    @log_time
    def write_tree_structure_to_file(self):
        """
        Write the tree structure to a file.
        """
        if "zarr" in self.file_store:
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file)
            elif self.file_store == "zarr_z":
                root = zarr.open(
                    zarr.storage.ZipStore(self.index_file, mode="a"), mode="a"
                )

            # /index_root
            root.create_group("index_root")
            # /index_root/embeddings (N,D) float16/float32
            root["index_root"].create_array(
                name="embeddings",
                shape=self.index["root"]["embeddings"].shape,
                dtype=self.index["root"]["embeddings"].dtype,
                chunks=self.chunk_size,
            )
            root["index_root"]["embeddings"][:] = self.index["root"]["embeddings"]
            # /index_root/node_ids (N,) uint32/uint64
            root["index_root"]["node_ids"] = np.array(self.index["root"]["node_ids"])

            def process_level(level_key, data):
                level = int(level_key.split("_")[1])
                # /lvl_{}
                root.create_group(level_key)
                for n, node in data.items():
                    # /lvl_{}/node_{}
                    root[level_key].create_group(n)
                    if level == self.levels - 1:
                        continue
                    # /lvl_{}/node_{}/embeddings
                    root[level_key][n].create_array(
                        name="embeddings",
                        shape=node["embeddings"].shape,
                        dtype=node["embeddings"].dtype,
                        chunks=self.chunk_size,
                    )
                    root[level_key][n]["embeddings"][:] = node["embeddings"]
                    # /lvl_{}/node_{}/node_ids|item_ids
                    root[level_key][n]["node_ids"] = np.array(
                        node["node_ids"], dtype=np.uint32
                    )
                    # /lvl_{}/node_{}/border
                    root[level_key][n]["border"] = np.array(
                        [node["border"][0], node["border"][1]]
                    )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.workers
            ) as executor:
                futures = [
                    executor.submit(process_level, k, v)
                    for k, v in self.index.items()
                    if k.startswith("lvl_")
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

        elif self.file_store == "h5":
            h5 = h5py.File(self.index_file, "a")
            # /index_root
            root_group = h5.create_group("index_root")
            # /index_root/embeddings
            root_group.create_dataset(
                "embeddings",
                data=self.index["root"]["embeddings"],
                maxshape=(None, self.index["root"]["embeddings"].shape[1]),
                chunks=True,
            )
            # /index_root/node_ids
            root_group.create_dataset(
                "node_ids",
                data=self.index["root"]["node_ids"],
                maxshape=(None,),
                chunks=True,
            )

            def process_level_h5(level_key, level_data):
                """
                Worker function to process a single level for HDF5.
                """
                level = int(level_key.split("_")[1])
                ids_key = "item_ids" if level == self.levels - 1 else "node_ids"
                lvl_group = h5.create_group(level_key)
                for node_key, node_data in level_data.items():
                    node_group = lvl_group.create_group(node_key)
                    node_group.create_dataset(
                        "embeddings",
                        data=node_data["embeddings"],
                        maxshape=(None, node_data["embeddings"].shape[1]),
                        chunks=True,
                    )
                    node_group.create_dataset(
                        ids_key,
                        data=node_data[ids_key],
                        maxshape=(None,),
                        chunks=True,
                    )
                    node_group.create_dataset("border", data=node_data["border"])

            # Use ThreadPoolExecutor to process levels in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_level_h5, level_key, level_data)
                    for level_key, level_data in self.index.items()
                    if level_key.startswith("lvl_")
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            h5.close()

    def determine_node_map(self, item_embeddings, offset=0):
        node_map = {}
        for idx, emb in tqdm(enumerate(item_embeddings)):
            top, distances = calculate_distances(
                emb, self.index["root"]["embeddings"], self.metric
            )
            lvl = "lvl_0"
            node = ""
            # Select the top non-empty node
            for t in top:
                node = f"node_{t}"
                if len(self.index[lvl][node]["node_ids"]) > 0:
                    break
            for l in range(self.levels):
                if l == self.levels - 1:
                    # If it is the bottom level add item with its distance into the node_map
                    if node in node_map:
                        node_map[node].append((offset + idx, distances[top[t]]))
                    else:
                        node_map[node] = [(offset + idx, distances[top[t]])]
                else:
                    top, distances = calculate_distances(
                        emb, self.index[lvl][node]["embeddings"], self.metric
                    )
                    curr_node = node
                    if l + 1 < self.levels - 1:
                        # Select the top non-empty node
                        for t, n in enumerate(top):
                            node = f"node_{self.index[lvl][curr_node]['node_ids'][n]}"
                            if len(self.index[f"lvl_{l+1}"][node]["node_ids"]) > 0:
                                break
                    else:
                        # Next level is the bottom, so t=0 as we do not check for empty
                        t = 0
                        node = f"node_{self.index[lvl][curr_node]['node_ids'][top[t]]}"
                    # Increment lvl
                    lvl = f"lvl_{l+1}"
        return node_map

    @log_time
    def write_cluster_items_to_file(self, clusters):
        """
        ### Parameters
        clusters: numpy.ndarray of cluster embeddings
        """
        if "zarr" in self.file_store:
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file)
            elif self.file_store == "zarr_z":
                root = zarr.open(
                    zarr.storage.ZipStore(self.index_file, mode="a"), mode="a"
                )

            lvl = f"lvl_{self.levels-1}"
            for c in clusters.keys():
                node = c
                # if node in root[lvl].keys():
                node_group = root[lvl][node]
                # /lvl_{}/node_{}/embeddings
                node_group.create_array(
                    name="embeddings",
                    shape=self.index[lvl][node]["embeddings"].shape,
                    dtype=self.index[lvl][node]["embeddings"].dtype,
                    chunks=self.chunk_size,
                )
                node_group["embeddings"][:] = self.index[lvl][node]["embeddings"][:]
                # /lvl_{}/node_{}/node_ids|item_ids
                node_group["item_ids"] = np.array(
                    self.index[lvl][node]["item_ids"], dtype=np.uint32
                )
                # /lvl_{}/node_{}/border
                node_group["border"] = np.array(
                    [
                        self.index[lvl][node]["border"][0],
                        self.index[lvl][node]["border"][1],
                    ]
                )
        elif self.file_store == "h5":
            h5 = h5py.File(self.index_file, "a")
            lvl = f"lvl_{self.levels-1}"
            for c in clusters.keys():
                node = c
                node_group = h5[lvl][node]
                old_size = node_group["embeddings"].shape[0]
                new_size = self.index[lvl][node]["embeddings"].shape[0]
                # Resize
                node_group["embeddings"].resize(
                    (new_size, node_group["embeddings"].shape[1])
                )
                # node_group["distances"].resize((new_size,))
                node_group["item_ids"].resize((new_size,))
                # Insert
                node_group["embeddings"][old_size:] = self.index[lvl][node][
                    "embeddings"
                ][old_size:]
                # node_group["distances"][old_size:] = self.index[lvl][node][
                #     "distances"
                # ][old_size:]
                node_group["item_ids"][old_size:] = self.index[lvl][node]["item_ids"][
                    old_size:
                ]
                node_group["border"][:] = self.index[lvl][node]["border"]
            h5.close()

    @log_time
    def write_items_batch(
        self,
        embeddings: zarr.Array | h5py.Dataset,
        all_indices: list[int],
        node_order: list[dict],
        clusters: dict,
        start_idx: int,
        end_idx: int,
    ):
        sorted_indices = sorted(all_indices)
        sort_map = {}
        for i in all_indices:
            sort_map[i] = sorted_indices.index(i)

        embs = embeddings[sorted_indices][...]
        del sorted_indices
        del all_indices

        self.logger.info("Updating in-memory index")
        lvl = f"lvl_{self.levels-1}"
        # cnt = 0
        for node in node_order:
            ids = clusters[node]["ids"]
            distances = clusters[node]["distances"]
            self.index[lvl][node]["item_ids"] = ids
            self.index[lvl][node]["distances"] = np.array(distances)
            self.index[lvl][node]["embeddings"] = np.array(
                [embs[sort_map[i]] for i in ids]
            )
            # TODO: Move the below part to a function and call it after index is built
            if self.metric == Metric.IP or self.metric == Metric.COS:
                if self.index[lvl][node]["border"] is None or self.index[lvl][node][
                    "border"
                ][1] > min(distances):
                    mn = np.argsort(distances)[0]
                    self.index[lvl][node]["border"] = (ids[mn], distances[mn])
            elif self.metric == Metric.L2:
                if self.index[lvl][node]["border"] is None or self.index[lvl][node][
                    "border"
                ][1] < max(distances):
                    mx = np.argsort(distances)[::-1][0]
                    self.index[lvl][node]["border"] = (ids[mx], distances[mx])
            self.align_specific_node(lvl, node)

        self.logger.info(f"Writing batch (cluster {start_idx} to {end_idx})")
        self.write_cluster_items_to_file(clusters=clusters)

    @log_time
    def add_items_concurrent(
        self,
        embeddings_file: Path,
        chunk_size=100000,
        grp=False,
        grp_name="embeddings",
    ):
        """
        Get a map corresponding to which cluster node they will end up in on the last level
        """
        if self.workers > cpu_count():
            processes = cpu_count() - 1
        else:
            processes = self.workers

        embeddings = None
        embeddings = get_source_embeddings(embeddings_file, grp, grp_name)

        chunk_size = chunk_size
        partial_node_maps = []
        total_items = embeddings.shape[0]
        # Determine the level-1 clusters of items in batches (chunk_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as executor:
            # Create a list of futures for each chunk
            futures = []
            # Iterate over the chunks of data
            for start_idx in range(0, total_items, chunk_size):
                # Calculate the end index for the current chunk
                end_idx = min(start_idx + chunk_size, total_items)
                # Get the chunk of data
                chunk_data = embeddings[start_idx:end_idx][...]
                # Submit the chunk to the executor for processing and append to the futures list
                futures.append(
                    executor.submit(self.determine_node_map, chunk_data, start_idx)
                )

            # Wait for all futures to complete and gather the results
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Gathering thread results (Determining node mappings)",
                total=len(futures),
            ):
                # Get the result from the future and append it to the list
                partial_node_maps.append(future.result())
        del chunk_data

        # This is the number of clusters to write per batch, it is set to 4 times the number of processes
        # The higher batch number is to avoid scenarios where single or a few processes deal with large clusters
        # This is not unavoidable, without adding extra logic to the code
        # but it is a good tradeoff between performance and simplicity
        clst_batches = int(self.total_clusters // (processes * 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as executor:
            # Create a list of futures for each batch
            futures = []
            # Iterate over the batches of data
            for start_idx in range(0, self.total_clusters, clst_batches):
                # Calculate the end index for the current batch
                end_idx = min(start_idx + clst_batches, self.total_clusters)
                clusters = {}
                for n in tqdm(range(start_idx, end_idx), desc="Preparing batch"):
                    # From the partial_node_maps, get the item ids and distances
                    # for the current cluster node
                    node = f"node_{n}"
                    for pmap in partial_node_maps:
                        # Check if the node exists in the partial_node_map
                        if node in pmap:
                            # If it does, get the item ids and distances
                            # and add them to the clusters dict
                            for e_idx, dist in pmap[node]:
                                if node not in clusters:
                                    clusters[node] = {"ids": [], "distances": []}
                                clusters[node]["ids"].append(e_idx)
                                clusters[node]["distances"].append(dist)

                node_order = []
                all_indices = []
                # Store the collection item ids of the clusters
                for n in clusters:
                    all_indices += clusters[n]["ids"]
                    node_order.append(n)

                if len(all_indices) == 0:
                    self.logger.info(
                        f"Skipping batch with no allocated embeddings ({len(all_indices)})"
                    )
                    continue
                else:
                    self.logger.info(
                        f"Writing cluster batch embeddings ({len(all_indices)})"
                    )

                    # Submit the batch to the executor for processing
                    futures.append(
                        executor.submit(
                            self.write_items_batch,
                            embeddings,
                            all_indices,
                            node_order,
                            clusters,
                            start_idx,
                            end_idx,
                        )
                    )
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Gathering thread results (Writing clusters)",
                total=len(futures),
            ):
                # Do nothing, just wait for all futures to complete
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in thread: {e}")
                    raise e
