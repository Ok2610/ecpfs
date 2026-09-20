import datetime

import numpy as np
import pytest

import ecpfs

from .conftest import build_two_clusters_index


@pytest.mark.parametrize(
    "call",
    [
        lambda index: index.set_memory_limit_bytes(1024),
        lambda index: index.new_search(
            query=np.array([0.0, 0.0], dtype=np.float32), k=1, search_exp=1, max_increments=-1, exclude_vec=[]
        ),
        lambda index: index.get_next_k_items(0, k=1, search_exp=1, max_increments=-1, exclude_vec=[]),
        lambda index: index.cleanup_persisted_queries_older_than(0.0),
        lambda index: index.insert(embeddings=np.array([[0.0, 0.0]], dtype=np.float32)),
    ],
)
def test_calling_a_method_after_close_raises_value_error(tmp_path, call):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)
    index.close()

    # A plain ValueError raised by PyO3, so pytest.raises is enough. The Rust
    # panics in test_errors.py need assert_is_panic_exception instead.
    with pytest.raises(ValueError, match="closed"):
        call(index)


def test_close_persists_a_buffered_query_for_a_fresh_index_to_resume(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    first, query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=1, search_exp=1, max_increments=-1, exclude_vec=[]
    )
    assert [item_id for _, item_id in first] == [0]
    index.close()

    reloaded = ecpfs.Index(index_path)
    second = reloaded.get_next_k_items(query_id, k=2, search_exp=1, max_increments=-1, exclude_vec=[])
    assert [item_id for _, item_id in second] == [1, 2]


def test_cleanup_persisted_queries_older_than_erases_stale_entries(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    _, query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=1, search_exp=1, max_increments=-1, exclude_vec=[]
    )
    index.close()

    reloaded = ecpfs.Index(index_path)
    assert reloaded.cleanup_persisted_queries_older_than(0.0) == 0, "nothing is older than the Unix epoch"

    # A cutoff built from a datetime rather than an hour count, since callers
    # convert their own dates to a timestamp.
    future_cutoff = (datetime.datetime.now() + datetime.timedelta(hours=1)).timestamp()
    erased = reloaded.cleanup_persisted_queries_older_than(future_cutoff)
    assert erased == 1

    resumed = reloaded.get_next_k_items(query_id, k=2, search_exp=1, max_increments=-1, exclude_vec=[])
    assert resumed == [], "cleanup must have erased the query, nothing left to resume"


def test_context_manager_closes_on_exit(tmp_path):
    index_path = build_two_clusters_index(tmp_path)

    with ecpfs.Index(index_path) as index:
        index.new_search(
            query=np.array([0.0, 0.0], dtype=np.float32), k=1, search_exp=1, max_increments=-1, exclude_vec=[]
        )

    # A second close is a no-op.
    index.close()
    with pytest.raises(ValueError, match="closed"):
        index.set_memory_limit_bytes(1024)


def test_context_manager_closes_even_when_the_block_raises(tmp_path):
    index_path = build_two_clusters_index(tmp_path)

    with pytest.raises(RuntimeError):
        with ecpfs.Index(index_path) as index:
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="closed"):
        index.set_memory_limit_bytes(1024)
