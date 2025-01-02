import math
import h5py
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
from pathlib import Path
from typing import Tuple
from math import sqrt
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import dill
from pathos.multiprocessing import Pool, cpu_count

class ECPBuilder:
    def __init__(self, levels: int, logger, target_cluster_size=100, metric="L2"):
        """
        Constructor.

        Parameters:

        levels: Number of levels in the index hierarchy
        target_cluster_size: Aim for clusters of this size (no guarantees)
        metric: Metric to use when building the index [L2 (Euclidean) | IP (Inner Product) | cos (Cosine Similarity)]
        """
        self.levels: int = levels
        self.target_cluster_size: int = target_cluster_size
        self.logger = logger
        self.metric: str = metric
        self.representative_ids: np.ndarray | None = None
        self.representative_embeddings: np.ndarray | None = None
        self.item_to_cluster_map = {}
        return

    def select_cluster_representatives(
        self,
        embeddings_file: Path,
        option="offset",
        save_to_file="",
        clst_ids_dsname="clst_item_ids",
        clst_emb_dsname="clst_embeddings"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the list of representatives for clusters.

        #### Parameters
        embeddings: Numpy 2D np.ndarray of embeddings

        option: "offset", "random", or "dissimilar".
        "offset" = select based of an offset determined through the index config info,
        "random" = arbitrarily select cluster representatives, and
        "dissimilar" = ensure that a representative is dissimilar from others at the same level.

        save_to_file: If a filename is specified, the selected representatives will be stored into an HDF5
        file with the specified name, using the dataset names from clst_emb_dsname and clst_ids_dsname.

        #### Returns
        A tuple of two numpy arrays, a 2D array for the embeddings, and a 1D array for their item ids.
        """
        with h5py.File(embeddings_file, "r") as f:
            total_items = len(f["embeddings"])
            self.total_clusters = math.ceil(total_items / self.target_cluster_size)
            self.node_size = math.ceil(self.total_clusters ** (1.0 / self.levels))

            if option == "offset":
                all_item_ids = np.arange(total_items).astype(np.uint32)
                self.representative_ids = all_item_ids[::self.target_cluster_size]
                self.logger.info(f"{len(self.representative_ids)} Cluster representatives selected")
                if len(self.representative_ids) != self.total_clusters:
                    self.logger.error(
                        "Number of representatives does not match the total clusters."
                    )
                self.logger.info(f"Getting representative embeddings from file")
                self.representative_embeddings = f["embeddings"][::self.target_cluster_size]
            elif option == "random":
                if self.total_clusters > total_items:
                    self.logger.error(
                        "Total clusters exceed the number of available embeddings."
                    )
                self.representative_ids = np.random.choice(
                    total_items, size=self.total_clusters, replace=False
                )
                self.logger.info(f"{len(self.representative_ids)} Cluster Representatives Selected")
                self.logger.info(f"Getting representative embeddings from file")
                self.representative_embeddings = f["embeddings"][self.representative_ids]
            elif option == "dissimilar":
                if self.metric == "IP":
                    """"""
                elif self.metric == "L2":
                    """"""
                elif self.metric == "cos":
                    """"""
                raise NotImplementedError()
            else:
                raise ValueError(
                    "Unknown option, the valid options are ['offset', 'random', 'dissimilar']"
                )

            if save_to_file != "":
                with h5py.File(save_to_file, "w") as hf:
                    hf[clst_ids_dsname] = self.representative_ids
                    hf[clst_emb_dsname] = self.representative_embeddings

        return self.representative_embeddings, self.representative_ids


    def get_cluster_representatives_from_file(
        self, fp: Path, emb_dsname="clst_embeddings", ids_dsname="clst_item_ids"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load cluster representatives from a HDF5 file.

        The HDF5 file should have two datasets, one for their representative embeddings and one for their item ids.

        #### Parameters
        emb_dsname: Dataset name of representative embeddings, default=clst_representatives
        ids_dsname: Dataset name of representative item ids, default=clst_item_ids

        #### Returns
        A tuple of two numpy arrays, a 2D array for the embeddings, and a 1D array for their item ids.
        """

        with h5py.File(fp, "r") as hf:
            self.representative_embeddings = hf[emb_dsname][:]
            self.representative_ids = hf[ids_dsname][:]

        return self.representative_embeddings, self.representative_ids


    def distance_root_node(self, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the distance to the embeddings of the root node

        #### Parameters
        emb: Embedding vector to calculate distance to root node embeddings

        #### Returns
        Output the argsorted top array and the calculated distance array
        """
        if self.metric == "IP":
            distances = np.dot(self.index["root"]["embeddings"], emb)
            top = np.argsort(distances)[::-1]
        elif self.metric == "L2":
            differences = self.index["root"]["embeddings"] - emb
            distances = np.linalg.norm(differences, axis=1)
            top = np.argsort(distances)
        elif self.metric == "cos":
            distances = cosine_similarity(self.index["root"]["embeddings"], (emb,)).flatten()
            top = np.argsort(distances)[::-1]
        return top, distances


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
        if self.metric == "IP":
            distances = np.dot(self.index[lvl][node]["embeddings"], emb)
            top = np.argsort(distances)[::-1]
        elif self.metric == "L2":
            differences = self.index[lvl][node]["embeddings"] - emb
            distances = np.linalg.norm(differences)
            top = np.argsort(distances)
        elif self.metric == "cos":
            distances = cosine_similarity(self.index[lvl][node]["embeddings"], (emb,)).flatten()
            top = np.argsort(distances)[::-1]
            return top, distances

    def align_specific_node(self, lvl, node):
        """
        Resize the embeddings and distances arrays of a node to match their actual size
        """
        self.index[lvl][node]["embeddings"].resize((
            len(self.index[lvl][node]["item_ids"]),
            self.index[lvl][node]["embeddings"].shape[1]
        ))
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
        if self.metric == "IP" or self.metric == "cos":
            new_border_dists = np.argsort(
                np.sort(self.index[lvl][node]["distances"])
            )
            self.index[lvl][node]["border"] = (
                new_border_dists[0],
                self.index[lvl][node]["distances"][new_border_dists[0]],
            )
        elif self.metric == "L2":
            new_border_dists = np.argsort(
                np.sort(self.index[lvl][node]["distances"])
            )[::-1]
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
                self.index[lvl][node]["embeddings"].resize((
                    len(self.index[lvl][node]["item_ids"]),
                    self.index[lvl][node]["embeddings"].shape[1]
                ))
                self.index[lvl][node]["distances"].resize(
                    len(self.index[lvl][node]["item_ids"])
                )
                if len(self.index[lvl][node]["item_ids"]) > 0:
                    self.update_node_border_info(lvl, node)
            if i+1 == self.levels - 1:
                lvl_range = self.total_clusters
            else:
                lvl_range = lvl_range * self.node_size


    def build_tree_h5(self, save_to_file="") -> None:
        """
        Build the hierarchical eCP index for an HDF5 file.

        #### Parameters
        save_to_file: Filename to store the index. If left blank nothing is stored to disk.
        """
        
        def check_to_replace_border(lvl: str, node: str, emb: np.ndarray, cl_idx: int, dist: float):
            """
            Check if the maximum number of items have been reached in a node, and 
            if the new item is better than the border item, replace and update border item info

            #### Parameters
            lvl: Level of the node
            node: The node to check
            emb: Embedding vector of item to check
            cl_idx: The index of the representative cluster
            """
            border_idx, border_dist = self.index[lvl][node]['border']
            if self.metric == "IP":
                if  border_dist < dist:
                    self.index[lvl][node]['embeddings'][border_idx] = emb
                    self.index[lvl][node]['items_ids'][border_idx] = self.representative_ids[cl_idx]
                    self.index[lvl][node]['node_ids'][border_idx] = cl_idx
                    self.index[lvl][node]['distances'][border_idx] = dist
                    update_node_border_info(lvl, node)
            elif self.metric == "L2":
                if border_dist > dist:
                    self.index[lvl][node]['embeddings'][border_idx] = emb
                    self.index[lvl][node]['items_ids'][border_idx] = self.representative_ids[cl_idx]
                    self.index[lvl][node]['node_ids'][border_idx] = cl_idx
                    self.index[lvl][node]['distances'][border_idx] = dist
                    update_node_border_info(lvl, node)


        def update_node_border_info(lvl: str, node: str):
            """
            Calculate and set the border item for the node at the provided level
            """
            # Update border info
            if self.metric == "IP":
                new_border_dists = np.argsort(np.sort(self.index['lvl_0'][node]['distances']))
                self.index[lvl][node]['border'] = \
                    (new_border_dists[0], self.index[lvl][node]['distances'][new_border_dists[0]])
            elif self.metric == "L2":
                new_border_dists = np.argsort(np.sort(self.index['lvl_0'][node]['distances']))[::-1]
                self.index[lvl][node]['border'] = \
                    (new_border_dists[0], self.index[lvl][node]['distances'][new_border_dists[0]])


        def align_node_embeddings_and_distances():
            """
            Resize the embeddings and distances arrays of all nodes to match their actual size
            """
            lvl_range = self.node_size
            for i in range(self.levels):
                lvl = 'lvl_' + str(i)
                self.index[lvl] = {}
                for j in range(lvl_range):
                    node = 'node_' + str(j)
                    self.index[lvl][node]['embeddings'].resize(len(self.index[lvl][node]['item_ids']))
                    self.index[lvl][node]['distances'].resize(len(self.index[lvl][node]['item_ids']))
                    update_node_border_info(lvl, node)
                if i == self.levels-1:
                    lvl_range = self.total_clusters
                else:
                    lvl_range = lvl_range * self.node_size


        if self.representative_embeddings is None or self.representative_ids is None:
            raise ValueError(
                """
                Cluster representatives not determined! 
                Set recompute to True, or call select_cluster_representatives(...) prior to calling build_tree(...)
                """
            )

        self.index = {}
        self.index['root']['embeddings'] = self.representative_embeddings[:self.node_size]
        self.index['root']['item_ids'] = self.representative_ids[:self.node_size]

        lvl_range = 1
        for l in range(self.levels):
            lvl_range = lvl_range * self.node_size
            if lvl_range > self.total_clusters:
                lvl_range = self.total_clusters
            lvl = 'lvl_' + str(l)
            self.index[lvl] = {}
            for i in range(lvl_range):
                node = 'node_' + str(i)
                self.index[lvl][node] = {
                    'embeddings': np.zeros(shape=(self.node_size, self.representative_embeddings.shape[1]),
                                           dtype=self.representative_embeddings.dtype),
                    'item_ids': [],
                    'node_ids': [],
                    'distances': np.zeros(shape=(self.node_size,), dtype=self.representative_embeddings.dtype),
                    'border': (0, None)
                }

        # Start building tree top-down
        # As we already have root node, we start by inserting the nodes of lvl_1 into lvl_0.
        # Then we input lvl_2 items into lvl_1, until we reach self.levels-1.
        lvl_range = self.node_size 
        for l in range(self.levels-1):
            lvl_range = lvl_range * self.node_size
            if lvl_range > self.total_clusters:
                lvl_range = self.total_clusters
            embeddings = self.representative_embeddings[:lvl_range]
            for cl_idx, emb in enumerate(embeddings):
                top, distances = self.distance_root_node(emb)
                n = top[0]
                curr_lvl = 0
                while True:
                    lvl = 'lvl_' + str(curr_lvl)
                    node = 'node_' + str(n)
                    if curr_lvl == l:
                        next = len(self.index[lvl][node]['item_ids'])
                        if next > self.index[lvl][node]['embeddings'].shape[0]:
                            concat_array = np.zeros(shape=(self.node_size, self.representative_embeddings.shape[1]),
                                                    dtype=self.representative_embeddings.dtype)
                            self.index[lvl][node]['embeddings'].concatenate(concat_array)
                            self.index[lvl][node]['distances'].concatenate(concat_array)
                        self.index[lvl][node]['embeddings'][next] = emb
                        self.index[lvl][node]['item_ids'].append(self.representative_ids[cl_idx])
                        self.index[lvl][node]['node_ids'].append(cl_idx)
                        self.index[lvl][node]['distances'][next] = distances[top[0]]
                        # TODO: Move the below part to a function and call it after index is built
                        if self.metric == "IP":
                            if self.index[lvl][node]['border'][1] is None \
                                or self.index[lvl][node]['border'][1] > distances[top[0]]:
                                self.index[lvl][node]['border'] = (next, distances[top[0]])
                        elif self.metric == "L2":
                            if self.index[lvl][node]['border'][1] is None \
                                or self.index[lvl][node]['border'][1] < distances[top[0]]:
                                self.index[lvl][node]['border'] = (next, distances[top[0]])
                        break
                    else:
                        top, distances = self.distance_level_node(emb, lvl, node)
                        n = self.index[lvl][node]['node_ids'][top[0]]
                    curr_lvl += 1
        
        # Resize the embeddings and distance arrays of nodes to actual size
        align_node_embeddings_and_distances()

        if save_to_file != "":
            with h5py.File(save_to_file, 'a') as h5:
                h5['index'] = self.index


    def add_items(self, item_embeddings: np.ndarray):
        """
        Add items into the index
        """
        for idx, emb in enumerate(item_embeddings):
            top, distances = self.distance_root_node(emb)
            n = top[0]
            for l in range(self.levels):
                lvl = 'lvl_' + str(l)
                node = 'node_' + str(n)
                if l == self.levels-1:
                    next = len(self.index[lvl][node]['item_ids'])
                    if next > self.index[lvl][node]['embeddings'].shape[0]:
                        concat_array = np.zeros(shape=(self.node_size, emb.shape[1]),
                                                dtype=emb.dtype)
                        self.index[lvl][node]['embeddings'].concatenate(concat_array)
                        self.index[lvl][node]['distances'].concatenate(concat_array)
                    self.index[lvl][node]['embeddings'][next] = emb
                    self.index[lvl][node]['item_ids'].append(idx)
                    self.index[lvl][node]['distances'][next] = distances[top[0]]
                    # TODO: Move the below part to a function and call it after index is built
                    if self.metric == "IP":
                        if self.index[lvl][node]['border'][1] is None \
                            or self.index[lvl][node]['border'][1] > distances[top[0]]:
                            self.index[lvl][node]['border'] = (next, distances[top[0]])
                    elif self.metric == "L2":
                        if self.index[lvl][node]['border'][1] is None \
                            or self.index[lvl][node]['border'][1] < distances[top[0]]:
                            self.index[lvl][node]['border'] = (next, distances[top[0]])
                else:
                    top, distances = self.distance_level_node(emb, lvl, node)
                    n = self.index[lvl][node]['node_ids'][top[0]]
    

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
                lvl = 'lvl_' + str(l)
                node = 'node_' + str(n)
                radius = self.index[lvl][node]['border'][1]
                top, distances = self.distance_level_node(query, lvl, node)
                for t in top:
                    self.item_pq.put((
                        distances[t],
                        self.index[lvl][node]['item_ids'][t]
                    ))
                leaf_cnt += 1
            else:
                # Keep popping from queue and adding the next level and node to the queue
                top, distances = self.distance_level_node(query, lvl, node)
                for t in top:
                    dist = distances[t] - radius if self.metric == "L2" else distances[t] + radius
                    if l+1 == self.levels-1:
                        self.tree_pq.put((
                            dist,
                            True, 
                            l+1,
                            self.index[lvl][node]['node_ids'][t]
                        ))
                    else:
                        self.tree_pq.put((
                            dist, 
                            False,
                            l+1, 
                            self.index[lvl][node]['node_ids'][t]
                        ))
        
        return [self.item_pq.get() for _ in range(k) if not self.item_pq.empty()]

