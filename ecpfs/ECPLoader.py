from typing import List, Tuple
import numpy as np


class Node:
    def __init__(self, node_size, dim, dtype=np.float16):
        self.embeddings: np.ndarray = np.zeros(shape=(node_size, dim), dtype=dtype)
        self.item_ids: List[np.uint32] = []
        self.node_ids: List[np.uint32] = []
        self.distances: np.ndarray = np.zeros(shape=(node_size,), dtype=dtype)
        self.border: Tuple[np.uint32, dtype] = (0, 0.0)
