import numpy as np
import pytest

import ecpfs

from .conftest import build_two_clusters_index


def test_memory_limit_bytes_requires_an_int_not_none(tmp_path):
    # The Python API has no way to ask for an unlimited cache, so None is a
    # type error rather than a way in.
    index_path = build_two_clusters_index(tmp_path)
    with pytest.raises(TypeError):
        ecpfs.Index(index_path, memory_limit_bytes=None)


def test_set_memory_limit_bytes_applies_without_a_reload_and_search_still_works(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    # Smaller than what the search below touches, so it must evict as it
    # goes, but large enough that a single node's own data still fits.
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
