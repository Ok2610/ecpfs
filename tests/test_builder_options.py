import json

import numpy as np
import pytest

import ecpfs

from .conftest import (
    TWO_CLUSTERS,
    TWO_CLUSTERS_UINT8,
    write_h5_embeddings,
    write_h5_embeddings_f16,
    write_h5_embeddings_uint8,
)


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


def test_default_embedding_dtype_matches_the_source_native_dtype(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings_f16(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)

    assert read_index_root_dtype(index_path) == "float16"


def test_explicit_embedding_dtype_downcasts_an_f32_source(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2, embedding_dtype=ecpfs.EmbeddingDtype.F16)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)

    assert read_index_root_dtype(index_path) == "float16"

    # Still readable and searchable after the downcast.
    index = ecpfs.Index(index_path)
    items, _query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    ids = sorted(item_id for _, item_id in items)
    assert ids == [0, 1, 2, 3, 4, 5, 6, 7]


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

def test_default_embedding_dtype_matches_a_uint8_source(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings_uint8(h5_path, TWO_CLUSTERS_UINT8)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)

    assert read_index_root_dtype(index_path) == "uint8"


def test_explicit_uint8_dtype_stores_narrow_and_still_searches(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS_UINT8.astype(np.float32))
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(
        index_path, levels=2, metric=ecpfs.Metric.L2, embedding_dtype=ecpfs.EmbeddingDtype.UInt8
    )
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)

    assert read_index_root_dtype(index_path) == "uint8"

    index = ecpfs.Index(index_path)
    items, _query_id = index.new_search(
        query=np.array([1.0, 1.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    ids = sorted(item_id for _, item_id in items)
    assert ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_int8_dtype_is_exposed_and_usable(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, np.array([[-100.0, -100.0], [-98.0, -98.0], [100.0, 100.0], [102.0, 102.0]], dtype=np.float32))
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(
        index_path, levels=1, metric=ecpfs.Metric.L2, embedding_dtype=ecpfs.EmbeddingDtype.Int8
    )
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)

    assert read_index_root_dtype(index_path) == "int8"
