import h5py
import numpy as np
import pytest

import ecpfs

from .conftest import TWO_CLUSTERS, build_two_clusters_index, write_h5_embeddings


def test_select_representatives_rejects_an_unsupported_embeddings_file_extension(tmp_path):
    index_path = tmp_path / "index.zarr"
    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)

    bogus_path = tmp_path / "embeddings.txt"
    bogus_path.write_text("not an embeddings file")

    with pytest.raises(ValueError, match="unsupported embeddings file format"):
        builder.select_representatives(bogus_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)


def test_select_representatives_rejects_an_unsupported_embeddings_dtype(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("embeddings", data=np.zeros((4, 2), dtype=np.int32))
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=1, metric=ecpfs.Metric.L2)
    with pytest.raises(ValueError, match="unsupported embeddings dtype"):
        builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)


def test_index_raises_runtime_error_for_a_directory_that_is_not_an_index(tmp_path):
    not_an_index = tmp_path / "not_an_index"
    not_an_index.mkdir()
    (not_an_index / "some_file.txt").write_text("hello")

    with pytest.raises(RuntimeError):
        ecpfs.Index(not_an_index)


def test_new_search_rejects_a_query_of_the_wrong_dimension(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    with pytest.raises(ValueError, match="dimension"):
        index.new_search(
            query=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            k=4,
            search_exp=4,
            max_increments=-1,
            exclude_vec=[],
        )


def test_build_before_select_representatives_is_a_runtime_error(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)
    with pytest.raises(RuntimeError, match="select_representatives"):
        builder.build(h5_path, fallback_batch_rows=100)
