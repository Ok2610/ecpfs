import ecpfs
from pathlib import Path
from zarr.storage import LocalStore
import zarr
import numpy as np

index_path = Path('../CLIP_Benchmark/data/LSC24/index/lsc_ecpfs_1MB.zarr')
store = LocalStore(index_path, read_only=True)
index_fp = zarr.open(store, mode="r")
root = index_fp["index_root"]["embeddings"][:].astype(np.float32)
a = ecpfs.ECPIndexRS(index_path)
# b = ecpfs.ECPIndexPy(index_path)

q = root[10]

items, qid = a.search(q.astype(np.float32), 100, search_exp=32)
items, qid = a.search(q.astype(np.float32), 100, search_exp=32, exclude=set([4550]))
print(items[:10])

# items = a.incremental_search(qid, 100, search_exp=32)
# print(items)

# items, qid = b.search(q, 100, search_exp=32, exclude_vec=[])
# print(items[:10])


