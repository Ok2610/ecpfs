import json

import pytest

import ecpfs

from .conftest import TWO_CLUSTERS, write_h5_embeddings


def read_index_root_dtype(index_path):
    meta_path = index_path / "index_root" / "embeddings" / "zarr.json"
    return json.loads(meta_path.read_text())["data_type"]


def read_index_root_chunk_shape(index_path):
    meta_path = index_path / "index_root" / "embeddings" / "zarr.json"
    return json.loads(meta_path.read_text())["chunk_grid"]["configuration"]["chunk_shape"]


def test_select_representatives_rejects_an_unknown_strategy(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)
    with pytest.raises(ValueError, match="offset"):
        builder.select_representatives(h5_path, target_cluster_items=2, strategy="bogus", fallback_batch_rows=100)


@pytest.mark.parametrize(
    ("embedding_dtype", "on_disk"),
    [
        (ecpfs.EmbeddingDtype.F32, "float32"),
        (ecpfs.EmbeddingDtype.F16, "float16"),
        (ecpfs.EmbeddingDtype.UInt8, "uint8"),
        (ecpfs.EmbeddingDtype.Int8, "int8"),
    ],
)
def test_each_embedding_dtype_reaches_the_on_disk_arrays(tmp_path, embedding_dtype, on_disk):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2, embedding_dtype=embedding_dtype)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)

    assert read_index_root_dtype(index_path) == on_disk


def test_max_chunk_bytes_is_threaded_through_to_the_on_disk_chunk_shape(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)

    default_index_path = tmp_path / "default.zarr"
    default_builder = ecpfs.Builder(default_index_path, levels=2, metric=ecpfs.Metric.L2)
    default_builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    default_builder.build(h5_path, fallback_batch_rows=100)

    small_index_path = tmp_path / "small.zarr"
    # 2 floats/vec * 4 bytes = 8 bytes/vec, so max_chunk_bytes=64 caps a
    # chunk at 8 vecs, far below the default 50 MiB chunk's row count.
    small_builder = ecpfs.Builder(small_index_path, levels=2, metric=ecpfs.Metric.L2, max_chunk_bytes=64)
    small_builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    small_builder.build(h5_path, fallback_batch_rows=100)

    default_rows = read_index_root_chunk_shape(default_index_path)[0]
    small_rows = read_index_root_chunk_shape(small_index_path)[0]
    assert small_rows == 8
    assert small_rows < default_rows
