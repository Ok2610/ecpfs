import math
import h5py
import zarr
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
from pathlib import Path
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

from pathos.multiprocessing import Pool, cpu_count


class Metric(Enum):
    L2 = 0
    IP = 1
    COS = 2


class ECPBuilder:
    def __init__(
        self,
        levels: int,
        logger,
        target_cluster_size=100,
        metric=Metric.L2,
        index_file="ecpfs_index.zarr",
        file_store="zarr_l",
    ):
        """
        Constructor.

        Parameters:

        levels: Number of levels in the index hierarchy
        target_cluster_size: Aim for clusters of this size (no guarantees)
        metric: Metric to use when building the index [L2 (Euclidean) | IP (Inner Product) | cos (Cosine Similarity)]

        index_file: If a filename is specified, all index related data will be stored in a store. Default behaviour is to store it using a zarr.storage.LocalStore under the name "ecpfs_index.zarr". Set this to empty string if you do not want to store the index.

        file_store: "zarr_l" (LocalStore), "zarr_z" (ZipStore), or "h5" (HDF5)
        The file format to store the representative cluster embeddings and ids. default="zarr_l"
        """
        self.levels: int = levels
        self.target_cluster_size: int = target_cluster_size
        self.logger = logger
        self.metric: str = metric
        self.index_file = index_file
        self.file_store = file_store
        self.representative_ids: np.ndarray | None = None
        self.representative_embeddings: np.ndarray | None = None
        # self.item_to_cluster_map = {}
        self.chunk_size = (-1, -1)
        return

    def select_cluster_representatives(
        self,
        embeddings_file: Path,
        option="offset",
        grp=False,
        grp_name="embeddings",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the list of representatives for clusters.

        #### Parameters
        embeddings_file: A .h5/.zarr/.zip/.zipstore file that stores 2D np.ndarray of embeddings (.zarr = LocalStore, .zip/.zipstore = ZipStore)

        option: "offset", "random", or "dissimilar".
        "offset" = select based of an offset determined through the index config info,
        "random" = arbitrarily select cluster representatives, and
        "dissimilar" = ensure that a representative is dissimilar from others at the same level.


        #### Returns
        A tuple of two numpy arrays, a 2D array for the embeddings, and a 1D array for their item ids.
        """
        embeddings = None
        emb_fp = None

        # Get embeddings from HDF5 or zarr
        if embeddings_file.suffix == ".h5":
            emb_fp = h5py.File(embeddings_file, "r")
            embeddings = emb_fp[grp_name]
        elif (
            embeddings_file.suffix == ".zarr"
        ):
            if grp:
                embeddings = zarr.open(embeddings_file, mode="r")[grp_name]
            else:
                embeddings = zarr.open(embeddings_file, mode="r")
        elif(embeddings_file.suffix == ".zip"
            or str.lower(embeddings_file.suffix) == ".zipstore"):
            emb_fp = zarr.storage.ZipStore(embeddings_file, mode="r")
            if grp:
                embeddings = zarr.open(emb_fp, mode="r")[grp_name]
            else:
                embeddings = zarr.open(emb_fp, mode="r")

        else:
            raise ValueError("Unknown embeddings file format")

        # Determine sizes
        total_items, total_features = embeddings.shape
        self.total_clusters = math.ceil(total_items / self.target_cluster_size)
        self.node_size = math.ceil(self.total_clusters ** (1.0 / self.levels))

        self.chunk_size = calculate_chunk_size(
            num_features=total_features,
            dtype=embeddings.dtype
        )

        # Select cluster centroids
        if option == "offset":
            all_item_ids = np.arange(total_items).astype(np.uint32)
            self.representative_ids = all_item_ids[:: self.target_cluster_size]
            self.logger.info(
                f"{len(self.representative_ids)} Cluster representatives selected"
            )
            if len(self.representative_ids) != self.total_clusters:
                self.logger.error(
                    "Number of representatives does not match the total clusters."
                )
                raise ValueError(f"Number of representatives does not match the total clusters. {total_items, len(self.representative_ids), self.total_clusters}")
            self.logger.info(f"Getting representative embeddings from file")
            self.representative_embeddings = embeddings[:: self.target_cluster_size]
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
            mbk = MiniBatchKMeans (
                init="k-means++", 
                n_clusters=self.total_clusters,
                batch_size=50000
            )
            for start in tqdm(range(0, embeddings.shape[0], 200000)):
                end = min(start+200000, embeddings.shape[0])
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
                "Unknown option, the valid options are ['offset', 'random', 'dissimilar']"
            )

        self.write_cluster_representatives()
        # Store the cluster representative embeddings and item ids
                # Close zarr stores if used
        if emb_fp is not None:
            emb_fp.close()

        return self.representative_embeddings, self.representative_ids


    def get_cluster_representatives_from_file(
        self,
        fp: Path,
        emb_dsname="clst_embeddings",
        ids_dsname="clst_item_ids",
        format="zarr_l",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load cluster representatives from a zarr (local or zip) store or a HDF5 file.

        The HDF5 file should have two datasets, one for their representative embeddings and one for their item ids.

        #### Parameters
        fp: File path
        emb_dsname: Dataset name of representative embeddings, default=clst_representatives
        ids_dsname: Dataset name of representative item ids, default=clst_item_ids
        format: "zarr_l" (LocalStore), "zarr_z" (ZipStore), "h5" (HDF5)

        #### Returns
        A tuple of two numpy arrays, a 2D array for the embeddings, and a 1D array for their item ids.
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
            dtype=self.representative_embeddings.dtype
        )

        self.write_cluster_representatives()

        return self.representative_embeddings, self.representative_ids

    def distance_root_node(self, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the distance to the embeddings of the root node

        #### Parameters
        emb: Embedding vector to calculate distance to root node embeddings

        #### Returns
        Output the argsorted top array and the calculated distance array
        """
        if self.metric == Metric.IP:
            distances = np.dot(self.index["root"]["embeddings"], emb)
            top = np.argsort(distances)[::-1]
        elif self.metric == Metric.L2:
            differences = self.index["root"]["embeddings"] - emb
            distances = np.linalg.norm(differences, axis=1)
            top = np.argsort(distances)
        elif self.metric == Metric.COS:
            distances = cosine_similarity(
                self.index["root"]["embeddings"], (emb,)
            ).flatten()
            top = np.argsort(distances)[::-1]
        return top, distances


    def write_index_info(self):
        if self.file_store == "zarr_l" or self.file_store == "zarr_z":
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file, mode='w')
            elif self.file_store == "zarr_z":
                root = zarr.open(zarr.storage.ZipStore(self.index_file, mode="w"), mode="w")
            root.create_group("info")
            root["info"]["levels"] = np.array([self.levels])
            # TODO: Check if zarr v3 supports strings
            root["info"]["metric"] = np.array([self.metric])
        elif self.file_store == "h5":
            with h5py.File(self.index_file, "a") as hf:
                hf.create_group("info")
                hf["info"]["levels"] = self.levels
                hf["info"]["metric"] = self.metric


    def write_cluster_representatives(self):
        # TODO: Once write_index_info goes in use change mode="w" to mode="a"
        clst_ids_dsname = "clst_item_ids"
        clst_emb_dsname = "clst_embeddings"
        if self.file_store == "zarr_l" or self.file_store == "zarr_z":
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file, mode='w')
            elif self.file_store == "zarr_z":
                root = zarr.open(zarr.storage.ZipStore(self.index_file, mode="w"), mode="w")

            root[clst_ids_dsname] = self.representative_ids
            root.create_array(
                name=clst_emb_dsname,
                shape=self.representative_embeddings.shape,
                dtype=self.representative_embeddings.dtype,
                chunks=self.chunk_size
            )
            root[clst_emb_dsname] = self.representative_embeddings
        elif self.file_store == "h5":
            with h5py.File(self.index_file, "w") as hf:
                hf[clst_ids_dsname] = self.representative_ids
                hf[clst_emb_dsname] = self.representative_embeddings


    def distance_level_node(
        self, emb: np.ndarray, lvl: str, node: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the distance to the embeddings of the node at the specified level

        #### Parameters
        emb: Embedding vector
        lvl: The level to check, ex. "lvl_0", "lvl_1"
        node: Node to check, ex. "node_0", "node_1"

        #### Returns
        Output the argsorted top array and the calculated distance array
        """
        if self.metric == Metric.IP:
            distances = np.dot(self.index[lvl][node]["embeddings"], emb)
            top = np.argsort(distances)[::-1]
        elif self.metric == Metric.L2:
            differences = self.index[lvl][node]["embeddings"] - emb
            distances = np.linalg.norm(differences)
            top = np.argsort(distances)
        elif self.metric == Metric.COS:
            distances = cosine_similarity(
                self.index[lvl][node]["embeddings"], (emb,)
            ).flatten()
            top = np.argsort(distances)[::-1]
        return top, distances

    def align_specific_node(self, lvl, node):
        """
        Resize the embeddings and distances arrays of a node to match their actual size
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
        # TODO
        # if len(self.index[lvl][node]["item_ids"]) > 0:
        #     update_node_border_info(lvl, node)

    def update_node_border_info(self, lvl: str, node: str):
        """
        Calculate and set the border item for the node at the provided level
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
            for j in range(lvl_range):
                node = "node_" + str(j)
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
            if i + 1 == self.levels - 1:
                lvl_range = self.total_clusters
            else:
                lvl_range = lvl_range * self.node_size

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

            #### Parameters
            lvl: Level of the node
            node: The node to check
            emb: Embedding vector of item to check
            cl_idx: The index of the representative cluster
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
        self.index["root"]["item_ids"] = self.representative_ids[: self.node_size]
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
            self.index[lvl] = {}
            for i in range(lvl_range):
                node = "node_" + str(i)
                self.index[lvl][node] = {
                    "embeddings": np.zeros(
                        shape=(self.node_size, self.representative_embeddings.shape[1]),
                        dtype=self.representative_embeddings.dtype,
                    ),
                    "item_ids": [],
                    "node_ids": [],
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
                top, distances = self.distance_root_node(emb)
                n = top[0]
                curr_lvl = 0
                while True:
                    lvl = "lvl_" + str(curr_lvl)
                    node = "node_" + str(n)
                    if curr_lvl == l:
                        next = len(self.index[lvl][node]["item_ids"])
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
                        self.index[lvl][node]["item_ids"].append(
                            self.representative_ids[cl_idx]
                        )
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
                        top, distances = self.distance_level_node(emb, lvl, node)
                        n = self.index[lvl][node]["node_ids"][top[0]]
                    curr_lvl += 1

        self.logger.info("Aligning arrays...")
        # Resize the embeddings and distance arrays of nodes to actual size
        self.align_node_embeddings_and_distances()

        if "zarr" in self.file_store:
            self.logger.info("Saving tree to file...")
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file)
            elif self.file_store == "zarr_z":
                root = zarr.open(zarr.storage.ZipStore(self.index_file, mode="a"), mode="a")

            # /index_root
            root.create_group("index_root")
            # /index_root/embeddings (N,D) float16/float32
            root["index_root"].create_array(
                name="embeddings",
                shape=self.index["root"]["embeddings"].shape,
                dtype=self.index["root"]["embeddings"].dtype,
                chunks=self.chunk_size
            )
            root["index_root"]["embeddings"] = self.index["root"]["embeddings"]
            # /index_root/item_ids (N,) uint32/uint64
            root["index_root"]["item_ids"] = np.array(self.index["root"]["item_ids"])
            for k, v in self.index.items():
                if not k.startswith("lvl_"):
                    continue
                # /lvl_{}
                root.create_group(k)
                for n, node in v.items():
                    # /lvl_{}/node_{}
                    root[k].create_group(n)
                    # /lvl_{}/node_{}/embeddings
                    root[k][n].create_array(
                        name="embeddings",
                        shape=node["embeddings"].shape,
                        dtype=node["embeddings"].dtype,
                        chunks=self.chunk_size
                    )
                    root[k][n]["embeddings"] = node["embeddings"]
                    # /lvl_{}/node_{}/item_ids
                    root[k][n]["item_ids"] = np.array(node["item_ids"], dtype=np.uint32)
                    # /lvl_{}/node_{}/node_ids
                    root[k][n]["node_ids"] = np.array(node["node_ids"], dtype=np.uint32)
                    # /lvl_{}/node_{}/distances
                    root[k][n]["distances"] = node["distances"]
                    # /lvl_{}/node_{}/border
                    root[k][n]["border"] = np.array([
                        node["border"][0],
                        node["border"][1]
                    ])
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
            # /index_root/item_ids
            root_group.create_dataset(
                "item_ids",
                data=self.index["root"]["item_ids"],
                maxshape=(None,),
                chunks=True,
            )
            for k, v in self.index.items():
                if not k.startswith("lvl_"):
                    continue
                # /lvl_{}
                lvl_group = h5.create_group(k)
                for n, node in v.items():
                    # /lvl_{}/node_{}
                    node_group = lvl_group.create_group(n)
                    # /lvl_{}/node_{}/embeddings
                    node_group.create_dataset(
                        "embeddings",
                        data=node["embeddings"],
                        maxshape=(None, node["embeddings"].shape[1]),
                        chunks=True,
                    )
                    # /lvl_{}/node_{}/distances
                    node_group.create_dataset(
                        "distances",
                        data=node["distances"],
                        maxshape=(None,),
                        chunks=True,
                    )
                    # /lvl_{}/node_{}/item_ids
                    node_group.create_dataset(
                        "item_ids",
                        data=node["item_ids"],
                        maxshape=(None,),
                        chunks=True,
                    )
                    # /lvl_{}/node_{}/node_ids
                    node_group.create_dataset(
                        "node_ids",
                        data=node["node_ids"],
                        maxshape=(None,),
                        chunks=True,
                    )
                    # /lvl_{}/node_{}/border
                    node_group.create_dataset("border", data=node["border"])
            h5.close()
        self.logger.info("Done building tree!")

    def determine_node_map(self, item_embeddings, offset=0):
        node_map = {}
        for idx, emb in tqdm(enumerate(item_embeddings)):
            top, distances = self.distance_root_node(emb)
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
                    top, distances = self.distance_level_node(emb, lvl, node)
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

    def write_cluster_items_to_file(self, clusters):
        """
        ### Parameters
        clusters: numpy.ndarray of cluster embeddings
        """
        # TODO: Refactor this function into an ECPWriter class that has functions for the different formats
        if "zarr" in self.file_store:
            root = None
            if self.file_store == "zarr_l":
                root = zarr.open(self.index_file)
            elif self.file_store == "zarr_z":
                root = zarr.open(zarr.storage.ZipStore(self.index_file, mode="a"), mode="a")

            lvl = f"lvl_{self.levels-1}"
            if lvl not in root.keys():
                # /lvl_{self.levels-1}
                root.create_group(lvl)
                for c in tqdm(clusters.keys(), desc="Writing clusters to file"):
                    node = c
                    # /lvl_{self.levels-1}/node_{}
                    root[lvl].create_group(node)
                    # /lvl_{self.levels-1}/node_{}/embeddings
                    root[lvl][node].create_array(
                        name="embeddings",
                        shape=self.index[lvl][node]["embeddings"].shape,
                        dtype=self.index[lvl][node]["embeddings"].dtype,
                        chunks=self.chunk_size
                    )
                    root[lvl][node]["embeddings"] = self.index[lvl][node]["embeddings"]
                    # /lvl_{self.levels-1}/node_{}/distances
                    root[lvl][node]["distances"] = self.index[lvl][node]["distances"]
                    # /lvl_{self.levels-1}/node_{}/item_ids
                    root[lvl][node]["item_ids"] = np.array(self.index[lvl][node]["item_ids"], dtype=np.uint32)
                    # /lvl_{self.levels-1}/node_{}/border
                    root[lvl][node]["border"] = np.array([
                        self.index[lvl][node]["border"][0],
                        self.index[lvl][node]["border"][1]
                    ])
            else:
                for c in tqdm(clusters.keys(), desc="Writing clusters to file"):
                    node = c
                    if node in root[lvl].keys():
                        node_group = root[lvl][node]
                        old_size = node_group["embeddings"].shape[0]
                        new_size = self.index[lvl][node]["embeddings"].shape[0]
                        # Resize
                        node_group["embeddings"].resize(
                            (new_size, node_group["embeddings"].shape[1])
                        )
                        node_group["distances"].resize((new_size,))
                        node_group["item_ids"].resize((new_size,))
                        # Insert
                        node_group["embeddings"][old_size:] = self.index[lvl][node][
                            "embeddings"
                        ][old_size:]
                        node_group["distances"][old_size:] = self.index[lvl][node][
                            "distances"
                        ][old_size:]
                        node_group["item_ids"][old_size:] = self.index[lvl][node][
                            "item_ids"
                        ][old_size:]
                        node_group["border"] = np.array([
                            self.index[lvl][node]["border"][0],
                            self.index[lvl][node]["border"][0]
                        ])
                    else:
                        root[lvl].create_group(node)
                        # /lvl_{}/node_{}/embeddings
                        root[lvl][node].create_array(
                            name="embeddings",
                            shape=self.index[lvl][node]["embeddings"].shape,
                            dtype=self.index[lvl][node]["embeddings"].dtype,
                            chunks=self.chunk_size
                        )
                        root[lvl][node]["embeddings"] = self.index[lvl][node][
                            "embeddings"
                        ]
                        # /lvl_{}/node_{}/distances
                        root[lvl][node]["distances"] = self.index[lvl][node][
                            "distances"
                        ]
                        # /lvl_{}/node_{}/item_ids
                        root[lvl][node]["item_ids"] = np.array(self.index[lvl][node]["item_ids"], dtype=np.uint32)
                        # /lvl_{}/node_{}/border
                        root[lvl][node]["border"] = np.array([
                            self.index[lvl][node]["border"][0],
                            self.index[lvl][node]["border"][0]
                        ])
        elif self.file_store == "h5":
            h5 = h5py.File(self.index_file, "a")
            lvl = f"lvl_{self.levels-1}"
            if lvl not in h5.keys():
                lvl_group = h5.create_group(lvl)
                for c in tqdm(clusters.keys(), desc="Writing clusters to file"):
                    node = c
                    node_group = lvl_group.create_group(node)
                    node_group.create_dataset(
                        "embeddings",
                        data=self.index[lvl][node]["embeddings"],
                        maxshape=(None, self.index[lvl][node]["embeddings"].shape[1]),
                        chunks=True,
                    )
                    node_group.create_dataset(
                        "distances",
                        data=self.index[lvl][node]["distances"],
                        maxshape=(None,),
                        chunks=True,
                    )
                    node_group.create_dataset(
                        "item_ids",
                        data=self.index[lvl][node]["item_ids"],
                        maxshape=(None,),
                        chunks=True,
                    )
                    node_group.create_dataset(
                        "border", data=self.index[lvl][node]["border"]
                    )
            else:
                for c in tqdm(clusters.keys(), desc="Writing clusters to file"):
                    node = c
                    if node in h5[lvl].keys():
                        node_group = h5[lvl][node]
                        old_size = node_group["embeddings"].shape[0]
                        new_size = self.index[lvl][node]["embeddings"].shape[0]
                        # Resize
                        node_group["embeddings"].resize(
                            (new_size, node_group["embeddings"].shape[1])
                        )
                        node_group["distances"].resize((new_size,))
                        node_group["item_ids"].resize((new_size,))
                        # Insert
                        node_group["embeddings"][old_size:] = self.index[lvl][node][
                            "embeddings"
                        ][old_size:]
                        node_group["distances"][old_size:] = self.index[lvl][node][
                            "distances"
                        ][old_size:]
                        node_group["item_ids"][old_size:] = self.index[lvl][node][
                            "item_ids"
                        ][old_size:]
                        node_group["border"][:] = self.index[lvl][node]["border"]
                    else:
                        node_group = h5[lvl].create_group(node)
                        node_group.create_dataset(
                            "embeddings",
                            data=self.index[lvl][node]["embeddings"],
                            maxshape=(
                                None,
                                self.index[lvl][node]["embeddings"].shape[1],
                            ),
                            chunks=True,
                        )
                        node_group.create_dataset(
                            "distances",
                            data=self.index[lvl][node]["distances"],
                            maxshape=(None,),
                            chunks=True,
                        )
                        node_group.create_dataset(
                            "item_ids",
                            data=self.index[lvl][node]["item_ids"],
                            maxshape=(None,),
                            chunks=True,
                        )
                        node_group.create_dataset(
                            "border", data=self.index[lvl][node]["border"]
                        )
            h5.close()

    def add_items_concurrent(
        self,
        embeddings_file: Path,
        chunk_size=250000,
        workers=4,
        grp=False,
        grp_name="embeddings",
    ):
        """
        Get a map corresponding to which cluster node they will end up in on the last level
        """

        def process_chunk(item_embeddings, offset):
            """
            Worker function to process an embeddings chunk.
            For example, calls ecp.add_items_map on this chunk.
            """
            # ecp.add_items_map returns a dict where key is item id and value is (node, distance)
            return self.determine_node_map(
                item_embeddings=item_embeddings, offset=offset
            )

        if workers > cpu_count():
            processes = cpu_count() - 1
        else:
            processes = workers

        embeddings = None
        emb_fp = None
        if embeddings_file.suffix == ".h5":
            emb_fp = h5py.File(embeddings_file, "r")
            embeddings = emb_fp[grp_name]
        elif (
            embeddings_file.suffix == ".zarr"
        ):
            if grp:
                embeddings = zarr.open(embeddings_file, mode="r")[grp_name]
            else:
                embeddings = zarr.open(embeddings_file, mode="r")

        elif(
            embeddings_file.suffix == ".zip"
            or str.lower(embeddings_file.suffix) == ".zipstore"
        ):
            emb_fp = zarr.storage.ZipStore(embeddings_file, mode="r")
            if grp:
                embeddings = zarr.open(emb_fp, mode="r")[grp_name]
            else:
                embeddings = zarr.open(emb_fp, mode="r")

        chunk_size = chunk_size
        partial_node_maps = []
        total_items = embeddings.shape[0]
        # Determine the level-1 clusters of items in batches (chunk_size)
        with Pool(processes=processes) as pool:
            futures = []
            for start_idx in range(0, total_items, chunk_size):
                end_idx = min(start_idx + chunk_size, total_items)
                chunk_data = embeddings[start_idx:end_idx][...]
                async_results = pool.apply_async(
                    process_chunk, args=(chunk_data, start_idx)
                )
                futures.append(async_results)
            for future in tqdm(futures, desc="Gathering thread results"):
                partial_node_maps.append(future.get())
        del chunk_data

        clst_batches = int(self.total_clusters // 100)
        for start_idx in tqdm(
            range(0, self.total_clusters, clst_batches),
            desc="Writing clusters to file in batches",
        ):
            end_idx = min(start_idx + clst_batches, self.total_clusters)
            clusters = {}
            for n in tqdm(range(start_idx, end_idx), desc="Preparing batch"):
                node = f"node_{n}"
                for pmap in partial_node_maps:
                    if node in pmap:
                        for e_idx, dist in pmap[node]:
                            if node not in clusters:
                                clusters[node] = {'ids': [], 'distances': []}
                            clusters[node]['ids'].append(e_idx)
                            clusters[node]['distances'].append(dist)
                            # clusters[node].append((e_idx, dist))

            node_order = []
            all_indices = [] 
            # Store the collection item ids of the clusters
            for n in clusters:
                all_indices += clusters[n]['ids']
                node_order.append(n)

            if (len(all_indices) == 0):
                self.logger.info(
                    f"Skipping batch with no allocated embeddings ({len(all_indices)})"
                )
                continue
            else:
                self.logger.info(
                    f"Loading cluster batch embeddings ({len(all_indices)})"
                )
    
            sorted_indices = sorted(all_indices)
            embs = embeddings[sorted_indices][...]
            sort_map = {}
            for i in all_indices:
                sort_map[i] = sorted_indices.index(i)
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
                # TODO: Refactor so that it works
                # TODO: Move the below part to a function and call it after index is built
                if self.metric == "IP" or self.metric == "cos":
                    if (
                        self.index[lvl][node]["border"] is None
                        or self.index[lvl][node]["border"][1] > dist
                    ):
                        mn = np.argsort(distances)[0]
                        self.index[lvl][node]["border"] = (ids[mn], distances[mn])
                elif self.metric == "L2":
                    if (
                        self.index[lvl][node]["border"] is None
                        or self.index[lvl][node]["border"][1] < dist
                    ):
                        mx = np.argsort(distances)[::-1][0]
                        self.index[lvl][node]["border"] = (ids[mx], distances[mx])
                self.align_specific_node(lvl, node)

            self.logger.info(f"Writing batch (cluster {start_idx} to {end_idx})")
            self.write_cluster_items_to_file(
                clusters=clusters
            )

    def search_tree(self, query: np.ndarray, search_exp: int, k: int, restart=True):
        """
        Search the index tree and find the <search_exp> best leaf nodes.
        Uses priority queues to continue the previous search.

        #### Parameters:
        query: The query array
        search_exp: The amount of leaf nodes to explore
        k: Number of items to return
        restart: If the priority queue should be cleared or not

        #### Returns:
        A priority queue of items
        """
        leaf_cnt = 0
        if restart:
            self.tree_pq = PriorityQueue()
            self.item_pq = PriorityQueue()
        top, distances = self.distance_root_node(query)
        # Add to tree_pq
        for t in top:
            self.tree_pq.put((distances[t], False, 0, t))

        while leaf_cnt != search_exp and not self.tree_pq.empty():
            _, is_leaf, l, n = self.tree_pq.get()
            if is_leaf:
                # Pop leaf node items onto the item queue and increase the leaf_cnt
                lvl = "lvl_" + str(l)
                node = "node_" + str(n)
                radius = self.index[lvl][node]["border"][1]
                top, distances = self.distance_level_node(query, lvl, node)
                for t in top:
                    self.item_pq.put(
                        (distances[t], self.index[lvl][node]["item_ids"][t])
                    )
                leaf_cnt += 1
            else:
                # Keep popping from queue and adding the next level and node to the queue
                top, distances = self.distance_level_node(query, lvl, node)
                for t in top:
                    dist = (
                        distances[t] - radius
                        if self.metric == "L2"
                        else distances[t] + radius
                    )
                    if l + 1 == self.levels - 1:
                        self.tree_pq.put(
                            (dist, True, l + 1, self.index[lvl][node]["node_ids"][t])
                        )
                    else:
                        self.tree_pq.put(
                            (dist, False, l + 1, self.index[lvl][node]["node_ids"][t])
                        )

        return [self.item_pq.get() for _ in range(k) if not self.item_pq.empty()]
