from pathlib import Path
from queue import PriorityQueue
from typing import List, Tuple
import concurrent
import numpy as np
import zarr
from zarr.storage import LocalStore

from .ecp_node import ECPNode
from .utils import Metric, calculate_distances

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

    def __init__(self, index_path: Path, prefetch: int = 1, max_workers=4):
        """
        Initializes the ECPIndex with the given index path.
        Parameters:
            index_path (Path): Path to the index file.
            prefetch (int): Number of levels to prefetch (default=1, prefetch first level).
            max_workers (int): Number of threads to use for prefetching.
        """
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found (Path: {index_path})")
        store = LocalStore(index_path, read_only=True)
        index_fp = zarr.open(store, mode="r")
        self.root = index_fp["index_root"]["embeddings"][:]
        self.levels = index_fp["info"]["levels"][0]
        self.metric = Metric(index_fp["info"]["metric"][0])
        self.nodes = [[] for _ in range(self.levels)]
        for l in range(self.levels):
            lvl = f"lvl_{l+1}" # Index structure starts with lvl_1 and up
            lvl_nodes = sorted(
                [k for k in index_fp[lvl].keys() if "node" in k],
                key=lambda x: int(x.split('_')[1])
            )
            c_key = "node_ids" if l + 1 < self.levels else "item_ids"
            for node in lvl_nodes:
                self.nodes[l].append(ECPNode(node_fp=index_fp[lvl][node], c_key=c_key))
        if prefetch > 0:
            for i in range(prefetch):
                self.prefetch_level(i+1, max_workers=max_workers)

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

        nodes = self.nodes[level_index - 1] # since file starts lvl with 1

        print(f"Prefetching {len(nodes)} nodes from level {level_index}...")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        # Submit loading tasks for all nodes.
        _ = [executor.submit(load_node, node) for node in nodes]
        executor.shutdown(wait=False)

    def search(
        self,
        query: np.ndarray,
        k: int,
        search_exp: int = 64,
        exclude=set(),
        max_increments=-1
    ) -> Tuple[PriorityQueue, PriorityQueue]:
        """
        Initializes a new search for the k nearest neighbors of the query embedding.
        Parameters:
            query (np.ndarray): The query embedding.
            k (int): Number of nearest neighbors to find.
            search_exp (int): Number of leaf nodes to explore.
            exclude (set): A set of item ids to exclude from the search.
            max_increments (int): Number of times to expand scope, default=-1 (full).
        Returns:
            Tuple[PriorityQueue, PriorityQueue]: A tuple containing:
                - tree_pq: Priority queue for tree search.
                - item_pq: Priority queue for item search.
        """
        tree_pq = PriorityQueue()
        items = []
        self.incremental_search(
            query=query,
            tree_pq=tree_pq,
            items=items,
            k=k,
            search_exp=search_exp,
            exclude=exclude,
            max_increments=max_increments
        )
        return tree_pq, items

    def incremental_search(
        self, 
        query: np.ndarray, 
        tree_pq: PriorityQueue,
        items: List[Tuple[float, int]],
        k: int, 
        search_exp=64, 
        exclude=set(),
        max_increments=-1
    ) -> None:
        """
        Searches for the k nearest neighbors of the query embedding.
        Parameters:
            query (np.ndarray): The query embedding.
            k (int): Number of nearest neighbors to find.
            tree_pq: tree priority queue.
            item_pq: item priority queue.
            search_exp (int): Number of leaf nodes to explore.
            exclude (bool): A set of item ids to exclude from the search
            max_increments (int): Number of times to expand scope, default=-1 (full)
        Returns:
            Tuple[List[int], List[float]]: A tuple containing:
                - ids: The indices of the k nearest neighbors.
                - distances: The distances of the k nearest neighbors.
        """
        # Note: Since PriorityQueue, is a min priority,
        # if Metric.IP is used then we negate the distances
        # and negate them again when returning
        sign = -1 if self.metric == Metric.IP else 1
        items_append = items.append

        leaf_cnt = 0
        items_cnt = 0
        increments = 0
        if tree_pq.empty():
            root_top, root_distances = calculate_distances(query, self.root, self.metric)
            for t in root_top:
                tree_pq.put((sign * root_distances[t], False, 0, t))

        while not tree_pq.empty():
            _, is_leaf, lvl, node = tree_pq.get()
            if self.nodes[lvl][node].children is None:
                continue
            top, distances = calculate_distances(
                query, self.nodes[lvl][node].embeddings, self.metric
            )
            if is_leaf:
                # radius = self.index[lvl][node]["border"][1]
                # distances = distances.tolist()
                children = self.nodes[lvl][node].children
                for t in top:
                    item_id = children[t]
                    if item_id not in exclude:
                        items_append((sign * dist, item_id))
                        items_cnt += 1
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
                        tree_pq.put((sign * dist, True, lvl + 1, self.nodes[lvl][node].children[t]))
                    else:
                        tree_pq.put((sign * dist, False, lvl + 1, self.nodes[lvl][node].children[t]))

            if leaf_cnt == search_exp:
                if items_cnt >= k:
                    items = sorted(items, key=lambda x: x[0])
                    break
                if increments < max_increments or max_increments == -1:
                    increments += 1
                    search_exp *= 2
                else:
                    break

    def extract_k_items(self, k, items) -> Tuple[List[int], List[float]]:
        """
        Extracts the top k items from the item priority queue.
        Parameters:
            k (int): Number of items to extract.
            item_pq (PriorityQueue): The item priority queue.
        Returns:
            Tuple[List[int], List[float]]: A tuple containing:
                - ids: The indices of the k nearest neighbors.
                - scores: The distances of the k nearest neighbors.
        """
        sign = -1 if self.metric == Metric.IP else 1
        # top_k = [item_pq.get() for _ in range(k) if not item_pq.empty()]
        ids = [idx for _,idx in items[:k]]
        scores = [sign * score for score,_ in items[:k]] 
        items = items[k:]
        return ids, scores

