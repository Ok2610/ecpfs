import h5py
import numpy as np

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


def write_h5_embeddings_f16(path, embeddings, dataset_name="embeddings"):
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=np.asarray(embeddings, dtype=np.float16))
