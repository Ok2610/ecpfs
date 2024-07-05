import math
from typing import List, Tuple
import h5py
import numpy as np
from pathlib import Path

class Node:
    def __init__(self, node_size, dim, dtype=np.float16):
        self.embeddings: np.ndarray = np.zeros(shape=(node_size, dim), dtype=dtype)
        self.item_ids: List[np.uint32] = []
        self.node_ids: List[np.uint32] = []
        self.distances: np.ndarray = np.zeros(shape=(node_size,), dtype=dtype)
        self.border: Tuple[np.uint32, dtype | None] = (0, None)
 

class ECPBuilder:
    def __init__(self, levels: int, target_cluster_size=100, metric="L2"):
        self.levels : int = levels
        self.target_cluster_size : int = target_cluster_size
        self.metric : str = metric
        self.representative_ids : np.ndarray | None = None
        self.representative_embeddings : np.ndarray | None = None
        return  


    def select_cluster_representatives(
        self,
        embeddings: np.ndarray,
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
        self.total_clusters = math.ceil(len(embeddings)/self.target_cluster_size)
        self.node_size = math.ceil(self.total_clusters ** (1. / self.levels))

        if option == "offset":
            all_item_ids = np.arange(len(embeddings))
            self.representative_ids = all_item_ids[::self.target_cluster_size]
            if (len(self.representative_ids) != self.total_clusters):
                raise ValueError("Number of representatives does not match the total clusters.")
            self.representative_embeddings = embeddings[::self.target_cluster_size]
        elif option == "random":
            if self.total_clusters > len(embeddings):
                raise ValueError("Total clusters exceed the number of available embeddings.")
            self.representative_ids = np.random.choice(len(embeddings), size=self.total_clusters, replace=False)
            self.representative_embeddings = embeddings[self.representative_ids]
        elif option == "dissimilar":
            if self.metric == "IP":
                ""
            elif self.metric == "L2":
                ""
            raise NotImplementedError()
        else:
            raise ValueError("Unknown option, the valid options are ['offset', 'random', 'dissimilar']")

        if save_to_file != "":
            with h5py.File(save_to_file, 'w') as hf:
                hf[clst_ids_dsname] = self.representative_ids
                hf[clst_emb_dsname] = self.representative_embeddings

        return self.representative_embeddings, self.representative_ids 
 

    def get_cluster_representatives_from_file(
        self, fp: Path,
        emb_dsname="clst_embeddings",
        ids_dsname="clst_item_ids"
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

        with h5py.File(fp, 'r') as hf:
            self.representative_embeddings = hf[emb_dsname][:]
            self.representative_ids = hf[ids_dsname][:]
        
        return self.representative_embeddings, self.representative_ids
    

    def build_tree_h5(self, save_to_file="") -> None:
        """
        Build the hierarchical eCP index for an HDF5 file.

        #### Parameters
        save_to_file: Filename to store the index. If left blank nothing is stored to disk.
        """

        # Calculate the distance to the embeddings of the root node
        # Output the argsorted top array and the calculated distance array
        def distance_root_node(emb: np.ndarray) -> List:
            if self.metric == "IP":
                distances = np.dot(self.index['root']['embeddings'], emb)
                top = np.argsort(distances)[::-1]
            elif self.metric == "L2":
                differences = self.index['root']['embeddings'] - emb
                distances = np.linalg.norm(differences, axis=1)
                top = np.argsort(distances)
            return top, distances 


        # Calculate the distance to the embeddings of the node at the specified level
        # Output the argsorted top array and the calculated distance array
        def distance_level_node(emb: np.ndarray, lvl: str, node: str):
            if self.metric == "IP":
                distances =  np.dot(self.index[lvl][node]['embeddings'], emb)
                top = np.argsort(distances)[::-1]
            if self.metric == "L2":
                differences = self.index[lvl][node]['embeddings'] - emb
                distances = np.linalg.norm(differences)
                top = np.argsort(distances)
            return top, distances
        

        # Check if the maximum number of items are in a node, and if they are,
        # check if the new items is better than the border item
        def check_to_replace_border(lvl: str, node: str, emb: np.ndarray, cl_idx: int, dist: float):
            border_idx, border_dist = self.index[lvl][node]['border']
            if self.metric == "IP":
                if  border_dist < dist:
                    self.index[lvl][node]['embeddings'][border_idx] = emb
                    self.index[lvl][node]['items_ids'][border_idx] = self.representative_ids[cl_idx]
                    self.index[lvl][node]['cluster_ids'][border_idx] = cl_idx
                    self.index[lvl][node]['distances'][border_idx] = dist
                    update_node_border_info(lvl, node)
            elif self.metric == "L2":
                if border_dist > dist:
                    self.index[lvl][node]['embeddings'][border_idx] = emb
                    self.index[lvl][node]['items_ids'][border_idx] = self.representative_ids[cl_idx]
                    self.index[lvl][node]['cluster_ids'][border_idx] = cl_idx
                    self.index[lvl][node]['distances'][border_idx] = dist
                    update_node_border_info(lvl, node)


        # Calculate and set the border item for the node at the provided level
        def update_node_border_info(lvl: str, node: str):    
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

        lvl_range = self.node_size
        for i in range(self.levels):
            lvl = 'lvl_' + str(i)
            self.index[lvl] = {}
            for j in range(lvl_range):
                node = 'node_' + str(j)
                self.index[lvl][node] = {
                    'embeddings': np.zeros(shape=(self.node_size, self.representative_embeddings.shape[1]),
                                           dtype=self.representative_embeddings.dtype),
                    'item_ids': [],
                    'node_ids': [],
                    'distances': np.zeros(shape=(self.node_size,), dtype=self.representative_embeddings.dtype),
                    'border': (0, None)
                }
            if i == self.levels-1:
                lvl_range = self.total_clusters
            else:
                lvl_range = lvl_range * self.node_size
 

        # Start building tree top-down
        lvl_range = self.node_size 
        for l in range(self.levels):
            embeddings = self.representative_embeddings[:lvl_range]
            for cl_idx, emb in enumerate(embeddings):
                top, distances = distance_root_node(emb)
                curr_lvl = 0
                while True:
                    lvl = 'lvl_' + str(curr_lvl)
                    node = 'node_' + str(top[0]) # TODO: should be cluster_id?
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
                        top, distances = distance_level_node(emb, lvl, node)
                    curr_lvl += 1
            if l == self.levels-1:
                lvl_range = self.total_clusters
            else:
                lvl_range = lvl_range * self.node_size
        
        # Resize the embeddings and distance arrays of nodes to actual size
        align_node_embeddings_and_distances()

        if save_to_file != "":
            with h5py.File(save_to_file, 'a') as h5:
                h5['index'] = self.index


    def add_items():
        raise NotImplementedError()

