import threading

import numpy as np

import ecpfs

from .test_build_and_search import build_two_clusters_index


def test_insert_makes_a_new_item_findable_by_search(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    start_id, end_id = index.insert(embeddings=np.array([[0.05, 0.05]], dtype=np.float32))
    assert (start_id, end_id) == (8, 9)

    items, _ = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=9, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    ids = [item_id for _, item_id in items]
    assert 8 in ids


def test_insert_is_visible_to_a_freshly_opened_index_on_the_same_path(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)
    index.insert(embeddings=np.array([[0.05, 0.05]], dtype=np.float32))
    index.close()

    reloaded = ecpfs.Index(index_path)
    items, _ = reloaded.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=9, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    ids = [item_id for _, item_id in items]
    assert 8 in ids


def test_insert_multiple_points_in_one_call_are_all_findable(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    start_id, end_id = index.insert(
        embeddings=np.array([[0.05, 0.05], [10.05, 10.05]], dtype=np.float32),
    )
    assert (start_id, end_id) == (8, 10)

    near_origin, _ = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=9, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    assert 8 in [item_id for _, item_id in near_origin]

    near_far_cluster, _ = index.new_search(
        query=np.array([10.0, 10.0], dtype=np.float32), k=9, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    assert 9 in [item_id for _, item_id in near_far_cluster]


def test_concurrent_insert_and_search_from_multiple_threads(tmp_path):
    """Real threading.Thread's against one shared Index, releasing the GIL
    during both insert and search (see pyindex.rs's py.detach calls).
    This is what actually exercises ecp-core's per-leaf locking from
    Python, not just proves the Rust API accepts the calls."""
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)
    errors = []
    inserted_ids = []

    def search_repeatedly():
        try:
            for _ in range(50):
                index.new_search(
                    query=np.array([0.0, 0.0], dtype=np.float32),
                    k=10,
                    search_exp=4,
                    max_increments=-1,
                    exclude_vec=[],
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def insert_new_points():
        try:
            for i in range(10):
                start_id, _ = index.insert(embeddings=np.array([[0.01 + i * 0.001, 0.01]], dtype=np.float32))
                inserted_ids.append(start_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=search_repeatedly), threading.Thread(target=insert_new_points)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent insert/search raised: {errors}"
    assert len(inserted_ids) == 10

    items, _ = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=12, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    ids = {item_id for _, item_id in items}
    assert set(inserted_ids) <= ids, f"missing inserted items, got {ids}"
