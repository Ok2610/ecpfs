# ecpfs

An implementation of the extended Cluster Pruning (eCP) index.

The index format is stored in a versatile directory format, that enables flexible loading solutions in any language.

### Format

```
-> "collection" -> "name": str
-> "collection" -> "embeddings": np.ndarray, shape=(N, dim), dtype=ds.dtype

-> "index_info" -> "levels": int
-> "index_info" -> "metric": int (0=L2, 1=IP, 2=COS)
-> "rep_embeddings": np.ndarray, shape=((N/tcs), dim), dtype=ds.dtype
-> "rep_ids": np.ndarray, shape=((N/tcs),), dtype=np.uint32

-> "index_root" -> embeddings: np.ndarray, shape=(ns, dim), dtype=ds.dtype
-> "index_root" -> node_ids: np.ndarray, shape=(pow(ns),), dtype=np.uint32
-> "lvl_[1..L]" -> embeddings: np.ndarray, shape=(pow(ns,lvl), dim), dtype=ds.dtype
-> "lvl_[1..L-1]" -> node_ids: np.ndarray, shape=(pow(ns,lvl),), dtype=np.uint32
-> "lvl_L" -> item_ids: np.ndarray, shape=(pow(ns,lvl),), dtype=np.uint32
```

Currently, the collection group is not present, but will be an option in the future.

Can be further extended and modified.


### Still to be done

1. Add tests
2. Add missing documentation

