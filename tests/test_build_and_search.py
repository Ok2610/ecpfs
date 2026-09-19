import numpy as np

import ecpfs

from .conftest import TWO_CLUSTERS, write_h5_embeddings


def build_two_clusters_index(tmp_path, metric=ecpfs.Metric.L2, **builder_kwargs):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=metric, **builder_kwargs)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)
    return index_path


def test_round_trip_returns_nearest_items_in_ascending_score_order(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    items, query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[]
    )

    ids = [item_id for _, item_id in items]
    assert ids == [0, 1, 2, 3, 4, 5, 6, 7]

    scores = [score for score, _ in items]
    assert scores == sorted(scores)
    assert isinstance(query_id, int)


def test_get_next_k_items_resumes_the_same_query_without_dropping_or_repeating_items(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    first, query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=2, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    second = index.get_next_k_items(query_id=query_id, k=3, search_exp=4, max_increments=-1, exclude_vec=[])
    third = index.get_next_k_items(query_id=query_id, k=10, search_exp=4, max_increments=-1, exclude_vec=[])

    all_ids = [item_id for _, item_id in (*first, *second, *third)]
    assert all_ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_exclude_vec_omits_the_given_item_ids(tmp_path):
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    items, _query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[0, 1]
    )

    ids = [item_id for _, item_id in items]
    assert 0 not in ids
    assert 1 not in ids
    assert ids == [2, 3, 4, 5, 6, 7]


def test_ip_metric_ranks_highest_dot_product_first(tmp_path):
    # Unit-normalized vectors, since IP's own math ignores is_normalized
    # entirely. Normalizing is on the caller, not the index (see
    # ecp_core::utils::calculate_distances).
    angles = [0.0, 0.05, 0.10, 0.15, np.pi, np.pi + 0.05, np.pi + 0.10, np.pi + 0.15]
    unit_vectors = np.array([[np.cos(a), np.sin(a)] for a in angles], dtype=np.float32)

    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, unit_vectors)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.IP, is_normalized=True)
    builder.select_representatives(h5_path, target_cluster_items=2, strategy="offset", fallback_batch_rows=100)
    builder.build(h5_path, fallback_batch_rows=100)

    index = ecpfs.Index(index_path)
    items, _query_id = index.new_search(
        query=np.array([1.0, 0.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[]
    )

    assert items[0][1] == 0, "item 0 sits exactly on the query direction, so it must rank first"
    scores = [score for score, _ in items]
    assert scores == sorted(scores)


def test_search_exp_larger_than_the_real_tree_still_returns_sorted_results(tmp_path):
    # Regression for a bug (ecp-core, fixed 2026-09-11): incremental_search
    # only sorted results on the exit path where leaf_cnt exactly equals
    # search_exp, so an oversized search_exp (tree exhausted first) used to
    # return results in leaf-visit order instead of nearest-first.
    index_path = build_two_clusters_index(tmp_path)
    index = ecpfs.Index(index_path)

    items, _query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=8, search_exp=1000, max_increments=-1, exclude_vec=[]
    )

    ids = [item_id for _, item_id in items]
    assert ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_select_representatives_custom_bypasses_selection_but_build_still_needs_a_source_file(tmp_path):
    h5_path = tmp_path / "embeddings.h5"
    write_h5_embeddings(h5_path, TWO_CLUSTERS)
    index_path = tmp_path / "index.zarr"

    builder = ecpfs.Builder(index_path, levels=2, metric=ecpfs.Metric.L2)
    leader_ids = np.array([0, 2, 4, 6], dtype=np.uint32)
    leader_embeddings = TWO_CLUSTERS[leader_ids]
    builder.select_representatives_custom(leader_ids, leader_embeddings)
    builder.build(h5_path, fallback_batch_rows=100)

    index = ecpfs.Index(index_path)
    items, _query_id = index.new_search(
        query=np.array([0.0, 0.0], dtype=np.float32), k=8, search_exp=4, max_increments=-1, exclude_vec=[]
    )
    ids = sorted(item_id for _, item_id in items)
    assert ids == [0, 1, 2, 3, 4, 5, 6, 7]
