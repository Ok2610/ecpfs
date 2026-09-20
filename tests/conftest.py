import h5py
import numpy as np

import ecpfs

# Two well-separated clusters of 4, same geometry as ecp-core's
# `builder_produces_a_structure_that_searches_correctly` Rust test, so a
# nearest-to-farthest query from the origin has an unambiguous expected order.
TWO_CLUSTERS = np.array(
    [
        [0.0, 0.0],
        [0.4, 0.4],
        [1.0, 1.0],
        [1.4, 1.4],
        [10.0, 10.0],
        [10.4, 10.4],
        [11.0, 11.0],
        [11.4, 11.4],
    ],
    dtype=np.float32,
)


def write_h5_embeddings(path, embeddings, dataset_name="embeddings"):
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=np.asarray(embeddings, dtype=np.float32))


def build_two_clusters_index(tmp_path, metric=ecpfs.Metric.L2, **builder_kwargs):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=metric, **builder_kwargs)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)
    return index_path
