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

/// Regression test: `incremental_search` used to only sort `items` inside
/// the `leaf_cnt == search_exp` branch, so a search that instead ends by
/// running `tree_pq` dry (leaf_cnt never reaches search_exp because the
/// whole tree has fewer leaves than that) returned items in leaf-visit
/// order, not sorted by score. `search_exp=100` against a 4-leaf tree
/// forces exactly that exit path on every call.
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

/// With search_exp=1, the first pass only explores 1 leaf cluster (2 items:
/// ids 0,1), not enough for k=4. With max_increments=-1 (unlimited), the
/// retry path at query.rs's `leaf_cnt == search_exp` check must double
/// search_exp (1 -> 2) and keep going, exploring a 2nd cluster (ids 2,3) to
/// reach k. Never exercised before: every existing fixture used a search_exp
/// large enough to satisfy k on the first pass.
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

/// Same setup as the unlimited-retry test above, but with a *finite*
/// max_increments=1 (i.e. "at most 1 retry"), which should be just enough
/// to go from search_exp=1 to 2 and reach k=4, identically to the unlimited
/// case. This isolates the finite-counter comparison (`increments <
/// max_increments`) from the `max_increments == -1` special case, which is
/// the only branch the test above exercises.
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

/// Same fixture, but k=8 (all items) needs 2 doublings (search_exp 1 -> 2 -> 4)
/// to satisfy, while max_increments=1 permits only 1. new_search gets exactly
/// one incremental_search call with the caller's budget, then drains whatever
/// that found; it never gets a second, independent attempt. So it returns
/// fewer than k items (the 2 clusters/4 items reachable after the single
/// permitted retry), not all 8.
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

/// `Index.queries` holds every in-flight query, keyed by the `query_id`
/// `new_search` returns, so one query's `tree_pq`/`items` must never bleed
/// into another's. This test opens two queries against the *same* `Index` from opposite corners of the
/// fixture: query A from the origin (nearest-to-farthest: 0,1,2,3,4,5,6,7),
/// and query B from (11,11), exactly item 6's position (nearest-to-farthest:
/// 6,7,5,4,3,2,1,0). It interleaves `get_next_k_items` calls on both
/// `query_id`s, asserting each stream stays independent throughout.
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

    // In a levels=1 tree every popped node is a leaf, so leaf_cnt (what
    // search_exp actually counts) advances once per cluster regardless of
    // how many total node lookups that involves. search_exp=4 here means
    // "don't stop before all 4 clusters have been scanned", not "check 4
    // nodes" in general (those only coincide because there's nothing but
    // leaves in this particular tree).
    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

/// A `query_id` that was never created (or one that got evicted under
/// memory pressure) must yield an empty result, not a panic: a purely
/// memory-pressure-driven eviction shouldn't crash a caller that did
/// nothing wrong.
#[test]
fn missing_query_id_returns_empty_instead_of_panicking() {
    let index = build_test_index();

    let items = index.get_next_k_items(999, 4, 4, -1, &HashSet::new());
    assert!(items.is_empty());

    // A no-op, not a panic.
    index.incremental_search(999, 4, 4, -1, &HashSet::new());
}

/// Proves `Index` is actually thread-safe under `&self`, not just
/// API-compatible with concurrent callers: many threads run full searches
/// against one shared, warm `Index` at once and each must see the same
/// correct results a sequential caller would, with no panics, deadlocks,
/// or cross-query contamination.
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

    // BinaryHeap is a max-heap: highest score pops first, regardless of the
    // other fields (level/node_id/is_leaf take no part in ordering).
    assert_eq!(heap.pop().unwrap().node_id, 1);
    assert_eq!(heap.pop().unwrap().node_id, 7);
    assert_eq!(heap.pop().unwrap().node_id, 99);
}
