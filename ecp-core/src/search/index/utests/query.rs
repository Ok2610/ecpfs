use super::*;
use crate::search::index::fixtures::{
    build_ivf_style_index, build_test_index, build_three_level_test_index,
};
use ndarray::array;

#[test]
fn l2_search_returns_nearest_items_in_order() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    // search_exp=4 explores all 4 leaf nodes, so this is an exact top-4.
    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);

    let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
    assert!(
        scores.windows(2).all(|w| w[0] <= w[1]),
        "scores not sorted: {scores:?}"
    );
}

/// A search that ends by running `tree_pq` dry, before `search_exp` leaves
/// are explored, must still return sorted items. `search_exp=100` against a
/// 4-leaf tree forces that exit.
#[test]
fn results_are_sorted_even_when_the_tree_is_exhausted_before_search_exp_is_reached() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 8, 100, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn l2_search_respects_exclude_set() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];
    let exclude: HashSet<u32> = [0].into_iter().collect();

    let (items, _query_id) = index.new_search(query, 4, 4, -1, &exclude);

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![1, 2, 3, 4], "excluded item 0 must not appear");
}

/// k=1, search_exp=1 explores only the nearest leaf (items 0 and 1, both
/// pushed since a leaf is never partially processed) and drains 1, leaving
/// item 1 buffered.
#[test]
fn get_next_k_items_tops_up_a_partially_filled_buffer_below_k() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (first, query_id) = index.new_search(query, 1, 1, -1, &HashSet::new());
    assert_eq!(first.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![0]);

    let second = index.get_next_k_items(query_id, 4, 1, 2, &HashSet::new());
    assert_eq!(
        second.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
}

/// search_exp=1 explores 1 leaf (ids 0, 1), not enough for k=4. With
/// max_increments=-1, search_exp doubles to 2 and a 2nd leaf (ids 2, 3)
/// reaches k.
#[test]
fn search_exp_doubles_until_k_items_found_with_unlimited_retries() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 4, 1, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "doubling search_exp should eventually surface all 4 nearest items"
    );
}

/// As above, but max_increments=1: the one doubling it allows is exactly
/// enough for k=4. Covers the `increments < max_increments` check that -1
/// skips.
#[test]
fn finite_max_increments_still_allows_configured_number_of_retries() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 4, 1, 1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "max_increments=1 should permit the single retry needed to reach k=4"
    );
}

/// k=8 needs two doublings (search_exp 1 -> 2 -> 4) but max_increments=1
/// allows one, so new_search returns the 4 items from 2 leaves.
#[test]
fn new_search_stops_once_max_increments_is_exhausted() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 8, 1, 1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "should stop after its 1 permitted retry (2 clusters explored), not find all 8 items"
    );
}

/// Two queries on one `Index`, resumed in turn, must not share state.
/// A from the origin ranks items 0, 1, 2, 3, 4, 5, 6, 7; B from (11, 11),
/// item 6's position, ranks 6, 7, 5, 4, 3, 2, 1, 0.
#[test]
fn interleaved_queries_on_the_same_index_stay_independent() {
    let index = build_test_index();
    let query_a: Array1<f32> = array![0.0, 0.0];
    let query_b: Array1<f32> = array![11.0, 11.0];

    let (first_a, query_id_a) = index.new_search(query_a, 2, 4, -1, &HashSet::new());
    assert_eq!(
        first_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![0, 1]
    );

    let (first_b, query_id_b) = index.new_search(query_b, 2, 4, -1, &HashSet::new());
    assert_eq!(
        first_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![6, 7]
    );
    assert_ne!(query_id_a, query_id_b);

    // Interleaved: resume A, then B, then A again.
    let second_a = index.get_next_k_items(query_id_a, 2, 4, -1, &HashSet::new());
    assert_eq!(
        second_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![2, 3]
    );

    let second_b = index.get_next_k_items(query_id_b, 2, 4, -1, &HashSet::new());
    assert_eq!(
        second_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![5, 4]
    );

    let third_a = index.get_next_k_items(query_id_a, 2, 4, -1, &HashSet::new());
    assert_eq!(
        third_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![4, 5]
    );

    let third_b = index.get_next_k_items(query_id_b, 2, 4, -1, &HashSet::new());
    assert_eq!(
        third_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![3, 2]
    );
}

#[test]
fn three_level_tree_descends_through_intermediate_level() {
    let index = build_three_level_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    // search_exp=4 explores all 4 leaf nodes, so this is an exact top-4.
    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);

    let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
    assert!(
        scores.windows(2).all(|w| w[0] <= w[1]),
        "scores not sorted: {scores:?}"
    );
}

#[test]
fn levels_1_index_searches_like_ivf_without_panicking() {
    let index = build_ivf_style_index();
    let query: Array1<f32> = array![0.0, 0.0];

    // Every node here is a leaf, so search_exp=4 scans all 4 clusters.
    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

/// An unknown `query_id` yields no items, and searching it is a no-op.
#[test]
fn missing_query_id_returns_empty_instead_of_panicking() {
    let index = build_test_index();

    let items = index.get_next_k_items(999, 4, 4, -1, &HashSet::new());
    assert!(items.is_empty());

    // A no-op, not a panic.
    index.incremental_search(999, 4, 4, -1, &HashSet::new());
}

/// Many threads searching one shared `Index` each get the results a
/// sequential caller would.
#[test]
fn concurrent_searches_from_multiple_threads_return_correct_results() {
    let index = Arc::new(build_test_index());
    const THREADS: usize = 8;
    const SEARCHES_PER_THREAD: usize = 20;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let index = Arc::clone(&index);
            std::thread::spawn(move || {
                for _ in 0..SEARCHES_PER_THREAD {
                    let query: Array1<f32> = array![0.0, 0.0];
                    let (items, query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());
                    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
                    assert_eq!(ids, vec![0, 1, 2, 3]);

                    // search_exp=4 explores all 4 leaves, so all 8 items are
                    // already buffered; this page just drains the rest.
                    let more = index.get_next_k_items(query_id, 4, 4, -1, &HashSet::new());
                    let more_ids: Vec<u32> = more.iter().map(|(_, id)| *id).collect();
                    assert_eq!(more_ids, vec![4, 5, 6, 7]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }
}

#[test]
fn heap_entry_orders_by_score_only() {
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        score: NotNan::new(1.0).unwrap(),
        is_leaf: 0,
        level: 5,
        node_id: 99,
    });
    heap.push(HeapEntry {
        score: NotNan::new(3.0).unwrap(),
        is_leaf: 1,
        level: 0,
        node_id: 1,
    });
    heap.push(HeapEntry {
        score: NotNan::new(2.0).unwrap(),
        is_leaf: 0,
        level: 2,
        node_id: 7,
    });

    // Highest score pops first, regardless of the other fields.
    assert_eq!(heap.pop().unwrap().node_id, 1);
    assert_eq!(heap.pop().unwrap().node_id, 7);
    assert_eq!(heap.pop().unwrap().node_id, 99);
}
