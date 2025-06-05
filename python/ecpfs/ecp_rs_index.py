from ctypes import ArgumentError

from pathlib import Path
from typing import List, Tuple
import numpy as np
import zarr
from zarr.storage import LocalStore

from .utils import Metric
from ecpfs import ecp_index_rs

class ECPIndexRS():
    def __init__(self, index_path: Path, prefetch: int = 1, max_workers=4, dtype=None):
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
        root = index_fp["index_root"]["embeddings"][:].astype(np.float32)
        levels = index_fp["info"]["levels"][0]
        metric = Metric(index_fp["info"]["metric"][0])
        nodes = [[] for _ in range(levels)]
        for l in range(levels):
            lvl = f"lvl_{l+1}" # Index structure starts with lvl_1 and up
            lvl_nodes = sorted(
                [k for k in index_fp[lvl].keys() if "node" in k],
                key=lambda x: int(x.split('_')[1])
            )
            for node in lvl_nodes:
                nodes[l].append(f"/{lvl}/{node}") #ECPNode(node_fp=index_fp[lvl][node], c_key=c_key))

        self.index = ecp_index_rs.IndexWrapper(str(index_path), metric.name, levels, root, nodes)
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        search_exp: int = 64,
        exclude=set(),
        max_increments=-1
    ) -> Tuple[List[Tuple[float, int, int, int]], List[Tuple[float,int]]]:
        return self.index.new_search(query.astype(np.float32), k, search_exp, max_increments, list(exclude))
    

    def incremental_search(
        self, 
        query_id: np.ndarray, 
        k: int, 
        search_exp=64, 
        exclude=set(),
        max_increments=-1
    ) -> Tuple[List[Tuple[float, int, int, int]], List[Tuple[float,int]]]:
        return self.index.incremental_search(query_id, k, search_exp, max_increments, list(exclude))