from pathlib import Path
from queue import PriorityQueue
import time
from typing import List, Tuple
import concurrent
import numpy as np
import zarr
from zarr.storage import LocalStore

from ecpfs.utils import Metric, calculate_distances


class ECPIndex:
    """
    A class representing the eCP index for efficient nearest neighbor search.
    It uses a tree structure to organize the embeddings and allows for lazy loading of data.
    Attributes:
        root (zarr.Array): Root embeddings array.
        levels (int): Number of levels in the index tree.
        nodes (list): List of nodes at each level.
        tree_pq (PriorityQueue): Priority queue for tree search.
        item_pq (PriorityQueue): Priority queue for item search.
    """

    class _Node:
        """
        A class representing a node in the index tree.
        It lazily loads its embeddings and children.
        Attributes:
            node_fp (zarr.Group): Zarr group for the node.
            c_key (str): Key for children (node_ids | item_ids).
            embeddings (np.ndarray): Cached embeddings.
            children (np.ndarray): Cached children.
            last_access (float): Last access time.
            access_count (int): Access count.
        """

        def __init__(self, node_fp: zarr.Group, c_key: str):
            """
            Initializes the node with its zarr group and children key.
            Parameters:
                node_fp (zarr.Group): Zarr group for the node.
                c_key (str): Key for children (node_ids | item_ids).
            """
            self._node_fp = node_fp
            self._c_key = c_key
            self._embeddings = None
            self._children = None
            self._last_access = 0
            self._access_count = 0

        @property
        def embeddings(self):
            """
            This property will load and cache the embeddings of a node when accessed.
            Returns:
                np.ndarray: The embeddings of the node.
            """
            self._access_count += 1
            self._last_access = time.time()
            if self._embeddings is None:
                self._embeddings = self._node_fp["embeddings"][:]
            return self._embeddings

        @property
        def children(self):
            """
            This property will load and cache the children of a node when accessed.
            Returns:
                np.ndarray: The children of the node.
            """
            self._access_count += 1
            self._last_access = time.time()
            if self._children is None:
                self._children = self._node_fp[self._c_key][:]
            return self._children

        def clear_cache(self):
            """
            Clears the cached embeddings and children of the node.
            """
            self._embeddings = None
            self._children = None

        def is_loaded(self):
            """
            Checks if the node's embeddings or children are loaded.
            Returns:
                bool: True if either embeddings or children are loaded, False otherwise.
            """
            return self._embeddings is not None or self._children is not None

    def __init__(self, index_path: Path, prefetch: int = -1, max_workers=4):
        """
        Initializes the ECPIndex with the given index path.
        Parameters:
            index_path (Path): Path to the index file.
            prefetch (int): Number of levels to prefetch (-1 for no prefetching).
            max_workers (int): Number of threads to use for prefetching.
        """
        store = LocalStore(index_path)
        index_fp = zarr.open(store, mode="r")
        self.root = index_fp["index_root"]["embeddings"]
        self.levels = index_fp["info"]["levels"][0]
        self.metric = Metric(index_fp["info"]["metric"][0])
        self.nodes = [[] for _ in range(self.levels)]
        for i in range(self.levels):
            lvl = f"lvl_{i}"
            lvl_nodes = [k for k in index_fp[lvl].keys() if "node" in k]
            c_key = "node_ids" if i + 1 < self.levels else "item_ids"
            for node in lvl_nodes:
                self.nodes[i].append(
                    self._Node(node_fp=index_fp[lvl][node], c_key=c_key)
                )
        if prefetch > -1:
            for i in range(prefetch + 1):
                self.prefetch_level(i, max_workers=max_workers)

    def cleanup_cache(self, max_loaded_nodes: int):
        """
        If more than max_loaded_nodes have their arrays loaded,
        clears the cache for nodes that were accessed the least recently.
        Parameters:
            max_loaded_nodes (int): Maximum number of loaded nodes allowed.
        """
        loaded_nodes = []
        for level in self.nodes:
            for node in level:
                if node.is_loaded():
                    loaded_nodes.append(node)
        total_loaded = len(loaded_nodes)
        print(f"Total loaded nodes: {total_loaded}")
        if total_loaded <= max_loaded_nodes:
            return
        # Calculate how many node caches to clear.
        nodes_to_clear = total_loaded - max_loaded_nodes
        # Sort by last access (least recently accessed first).
        loaded_nodes.sort(key=lambda n: n._last_access)
        print(f"Clearing cache for {nodes_to_clear} nodes...")
        for node in loaded_nodes[:nodes_to_clear]:
            node.clear_cache()

    def loaded_nodes(self) -> int:
        """
        Returns the number of nodes that have their arrays loaded.
        """
        loaded_nodes = 0
        for level in self.nodes:
            for node in level:
                if node.is_loaded():
                    loaded_nodes += 1
        return loaded_nodes

    def prefetch_level(self, level_index: int, max_workers: int = 4):
        """
        Prefetch all nodes in a specified level in the background.
        Parameters:
            level_index (int): Index of the level to prefetch.
            max_workers (int): Number of threads to use for prefetching.
        """

        def load_node(node):
            # Access the properties to trigger the lazy load.
            _ = node.embeddings
            _ = node.children

        nodes = self.nodes[level_index]

        print(f"Prefetching {len(nodes)} nodes from level {level_index}...")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        # Submit loading tasks for all nodes.
        _ = [executor.submit(load_node, node) for node in nodes]
        executor.shutdown(wait=False)

    def search(
        self, query: np.ndarray, k: int, search_exp=64, restart=True
    ) -> Tuple[List[float], List[int]]:
        """
        Searches for the k nearest neighbors of the query embedding.
        Parameters:
            query (np.ndarray): The query embedding.
            k (int): Number of nearest neighbors to find.
            search_exp (int): Number of leaf nodes to explore.
            restart (bool): Whether to restart the search.
        Returns:
            Tuple[List[float], List[int]]: A tuple containing:
                - distances: The distances of the k nearest neighbors.
                - ids: The indices of the k nearest neighbors.
        """
        leaf_cnt = 0
        if restart:
            self.tree_pq = PriorityQueue()
            self.item_pq = PriorityQueue()
        root_top, root_distances = calculate_distances(query, self.root, self.metric)
        for t in root_top:
            self.tree_pq.put((root_distances[t], False, 0, t))

        while leaf_cnt != search_exp and not self.tree_pq.empty():
            _, is_leaf, lvl, node = self.tree_pq.get()
            top, distances = calculate_distances(
                query, self.nodes[lvl][node].embeddings, self.metric
            )
            if is_leaf:
                # radius = self.index[lvl][node]["border"][1]
                for t in top:
                    self.item_pq.put((distances[t], self.nodes[lvl][node].children[t]))
                leaf_cnt += 1
            else:
                # Keep popping from queue and adding the next level and node to the queue
                for t in top:
                    dist = (
                        distances[t]  # - radius
                        # if self.metric == Metric.L2
                        # else distances[t] + radius
                    )
                    if lvl + 1 == self.levels - 1:
                        self.tree_pq.put(
                            (dist, True, lvl + 1, self.nodes[lvl][node].children[t])
                        )
                    else:
                        self.tree_pq.put(
                            (dist, False, lvl + 1, self.nodes[lvl][node].children[t])
                        )
        top_k = sorted(
            [self.item_pq.get() for _ in range(k) if not self.item_pq.empty()],
            reverse=True,
        )
        return [t[0] for t in top_k], [t[1] for t in top_k]
