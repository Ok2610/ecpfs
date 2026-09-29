import numpy as np
import zarr

from .conftest import build_two_clusters_index


def test_zarr_python_can_open_and_walk_a_built_index(tmp_path):
    index_path = build_two_clusters_index(tmp_path)

    root = zarr.open_group(index_path, mode="r")
    assert set(root.keys()) >= {
        "rep_embeddings",
        "rep_item_ids",
        "info",
        "index_root",
        "lvl_1",
        "lvl_2",
    }

    root_embeddings = root["index_root/embeddings"]
    assert root_embeddings.shape[1] == 2

    node_0 = root["lvl_1"]["node_0"]
    assert "embeddings" in node_0

    # Values read through group navigation must match a direct full-path
    # read, confirming groups didn't change what the data actually is.
    via_group = node_0["embeddings"][:]
    via_full_path = root["lvl_1/node_0/embeddings"][:]
    assert via_group.dtype == np.float32
    assert np.array_equal(via_group, via_full_path)
