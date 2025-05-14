import zarr
import time


class ECPNode:
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
