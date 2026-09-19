import numpy as np
import pytest

import ecpfs

from .conftest import TWO_CLUSTERS, write_h5_embeddings


def build_two_clusters_index(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)
    return index_path


def test_memory_limit_bytes_requires_an_int_not_none(tmp_path):
    # Regression for a fix (ecp-core audit, 2026-09-10): the old
    # None-means-cache-forever option was deliberately removed from both
    # Builder and Index's public entry points.
    index_path = build_two_clusters_index(tmp_path)
    with pytest.raises(TypeError):
        ecpfs.Index(index_path, memory_limit_bytes=None)


def test_index_and_builder_accept_the_default_memory_limit_with_no_argument(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)
    items, _query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    assert len(items) == 8


def test_set_memory_limit_bytes_applies_without_a_reload_and_search_still_works(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    # Small enough to force eviction of at least one already-touched node,
    # but large enough that a single node's own data still fits.
    index.set_memory_limit_bytes(64)

    items, _query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    ids = [item_id for _, item_id in items]
    assert ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_set_memory_limit_bytes_requires_an_int_not_none(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)
    with pytest.raises(TypeError):
        index.set_memory_limit_bytes(None)
