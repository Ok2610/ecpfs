# ecpfs

An implementation of the extended Cluster Pruning (eCP) index.

The index format is stored in a versatile directory format, that enables flexible loading solutions in any language.

### Format

```
"collection" -> "name": str
"collection" -> "embeddings": np.ndarray, shape=(N, dim)

"index" -> "target_cluster_size": number of items to aim for in each cluster
"index" -> "total_clusters": N / target_cluster_size
"index" -> "node_size": total_clusters ** pow(1./L)
"index" -> "levels": height of tree
"index" -> "representetive_embeddings": Cluster leaders, np.ndarray
"index" -> "representative_ids": Item ids of the cluster leaders, List[int]
"index" -> "representative_option": How the representatives were selected, ["offset", "random", "dissimilar"]

"index" -> "root": Node for top level containing total_clusters ** pow(1./L) embeddings
"index" -> "lvl_[0..L-1]" -> "node_id"
"index" -> "leafs": Clusters, similar to Node but no cluster_ids is set

"node_id": {
	"embeddings": np.ndarray, shape=(node_size, dim)
	"item_ids": List[int],
	"node_ids": List[int],
	"border": Tuple[int, dtype]
}

```

Can be further extended and modified.
