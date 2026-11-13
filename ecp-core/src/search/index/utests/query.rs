use super::*;
use crate::search::index::fixtures::{
    build_ivf_style_index, build_test_index, build_three_level_test_index,
};
use crate::test_fixtures::{
    as_readable_writable_listable, new_memory_store, write_index_info, write_index_root,
    write_item_counts, write_node,
};
use ndarray::array;

#[test]
fn l2_search_returns_nearest_items_in_order() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    // search_exp=4 explores all 4 leaf nodes, so this is an exact top-4.
    let (items, _query_id) = index
        .new_search(query, 4, 4, -1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);

    let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
    assert!(
        scores.windows(2).all(|w| w[0] <= w[1]),
        "scores not sorted: {scores:?}"
    );
}

/// A search that ends by running `frontier` dry, before `search_exp` leaves
/// are explored, must still return sorted items. `search_exp=100` against a
/// 4-leaf tree forces that exit.
#[test]
fn results_are_sorted_even_when_the_tree_is_exhausted_before_search_exp_is_reached() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index
        .new_search(
            query,
            8,
            100,
            -1,
            &HashSet::new(),
            SearchStrategy::BestFirst,
        )
        .unwrap();

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn l2_search_respects_exclude_set() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];
    let exclude: HashSet<u32> = [0].into_iter().collect();

    let (items, _query_id) = index
        .new_search(query, 4, 4, -1, &exclude, SearchStrategy::BestFirst)
        .unwrap();

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![1, 2, 3, 4], "excluded item 0 must not appear");
}

/// k=1, search_exp=1 scores only the nearest leaf, buffering both its items
/// (0 and 1), and returns 1, leaving item 1 buffered.
#[test]
fn get_next_k_items_tops_up_a_partially_filled_buffer_below_k() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (first, query_id) = index
        .new_search(query, 1, 1, -1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();
    assert_eq!(first.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![0]);

    let second = index
        .get_next_k_items(query_id, 4, 1, 2, &HashSet::new())
        .unwrap();
    assert_eq!(
        second.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
}

/// new_search excludes item 2, which sits in leaf (2, 3), and scans only leaf (0, 1).
/// The resume scans leaf (2, 3) with an empty exclude, so item 2 stays out only if
/// the query kept new_search's exclude.
#[test]
fn a_resumed_query_keeps_excluding_ids_given_to_new_search() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];
    let exclude: HashSet<u32> = [2].into_iter().collect();
    let (_, query_id) = index
        .new_search(query, 1, 1, -1, &exclude, SearchStrategy::BestFirst)
        .unwrap();

    let next = index
        .get_next_k_items(query_id, 3, 1, -1, &HashSet::new())
        .unwrap();

    assert_eq!(
        next.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![1, 3, 4]
    );
}

/// new_search scans leaf (0, 1), adds items 0 and 1 to `items`, and returns item 0.
/// get_next_k_items then excludes item 1, which must remove it from `items`.
#[test]
fn an_id_excluded_on_a_later_call_is_removed_from_items() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = index
        .new_search(query, 1, 1, -1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();
    let exclude: HashSet<u32> = [1].into_iter().collect();

    let next = index
        .get_next_k_items(query_id, 2, 1, -1, &exclude)
        .unwrap();

    assert_eq!(
        next.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

/// search_exp=1 explores 1 leaf (ids 0, 1), not enough for k=4. With
/// max_increments=-1, search_exp doubles to 2 and a 2nd leaf (ids 2, 3)
/// reaches k.
#[test]
fn search_exp_doubles_until_k_items_found_with_unlimited_retries() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index
        .new_search(query, 4, 1, -1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();

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

    let (items, _query_id) = index
        .new_search(query, 4, 1, 1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();

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

    let (items, _query_id) = index
        .new_search(query, 8, 1, 1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();

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

    let (first_a, query_id_a) = index
        .new_search(
            query_a,
            2,
            4,
            -1,
            &HashSet::new(),
            SearchStrategy::BestFirst,
        )
        .unwrap();
    assert_eq!(
        first_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![0, 1]
    );

    let (first_b, query_id_b) = index
        .new_search(
            query_b,
            2,
            4,
            -1,
            &HashSet::new(),
            SearchStrategy::BestFirst,
        )
        .unwrap();
    assert_eq!(
        first_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![6, 7]
    );
    assert_ne!(query_id_a, query_id_b);

    // Resume A, then B, then A again.
    let second_a = index
        .get_next_k_items(query_id_a, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        second_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![2, 3]
    );

    let second_b = index
        .get_next_k_items(query_id_b, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        second_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![5, 4]
    );

    let third_a = index
        .get_next_k_items(query_id_a, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        third_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![4, 5]
    );

    let third_b = index
        .get_next_k_items(query_id_b, 2, 4, -1, &HashSet::new())
        .unwrap();
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
    let (items, _query_id) = index
        .new_search(query, 4, 4, -1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();

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
    let (items, _query_id) = index
        .new_search(query, 4, 4, -1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

/// An unknown `query_id` yields no items, and searching it is a no-op.
#[test]
fn missing_query_id_returns_empty_instead_of_panicking() {
    let index = build_test_index();

    let items = index
        .get_next_k_items(999, 4, 4, -1, &HashSet::new())
        .unwrap();
    assert!(items.is_empty());

    // A no-op, not a panic.
    index.incremental_search(999, 4, 4, -1).unwrap();
}

/// `search_exp=4` against this 4-leaf fixture visits every leaf in one call.
#[test]
fn incremental_search_returns_how_many_leaves_it_visited() {
    let index = build_test_index();
    let query_id = 1;
    index.queries.insert(
        query_id,
        Arc::new(Mutex::new(QueryState {
            query: array![0.0, 0.0],
            frontier: Frontier::BestFirst(BinaryHeap::new()),
            items: Vec::new(),
            exclude: HashSet::new(),
        })),
    );

    let leaf_nodes_scanned = index.incremental_search(query_id, 4, 4, -1).unwrap();
    assert_eq!(leaf_nodes_scanned, 4);
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
                    let (items, query_id) = index
                        .new_search(query, 4, 4, -1, &HashSet::new(), SearchStrategy::BestFirst)
                        .unwrap();
                    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
                    assert_eq!(ids, vec![0, 1, 2, 3]);

                    // search_exp=4 explores all 4 leaves, so all 8 items are
                    // already buffered; this page just drains the rest.
                    let more = index
                        .get_next_k_items(query_id, 4, 4, -1, &HashSet::new())
                        .unwrap();
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

/// Returns the item ids in `items`, in order.
fn ids(items: &ScoredItems) -> Vec<u32> {
    items.iter().map(|(_, id)| *id).collect()
}

/// Returns the node ids queued in each level's heap of LevelBeam query `query_id`, sorted.
fn level_heap_node_ids(index: &Index, query_id: usize) -> Vec<Vec<u32>> {
    let state_arc = index.queries.get(&query_id).unwrap();
    let state = state_arc.lock().unwrap();
    let Frontier::LevelBeam(heaps) = &state.frontier else {
        panic!("query {query_id} is not a LevelBeam query");
    };
    heaps
        .iter()
        .map(|heap| {
            let mut node_ids: Vec<u32> = heap.iter().map(|e| e.node_id).collect();
            node_ids.sort();
            node_ids
        })
        .collect()
}

/// With `search_exp` at least every level's node count, a descent visits every node, so
/// LevelBeam finds the same items as BestFirst.
#[test]
fn level_beam_matches_best_first_when_search_exp_covers_every_level() {
    for (index, k) in [(build_test_index(), 8), (build_three_level_test_index(), 4)] {
        let search = |strategy| {
            let query: Array1<f32> = array![0.0, 0.0];
            let (items, _) = index
                .new_search(query, k, 4, 0, &HashSet::new(), strategy)
                .unwrap();
            ids(&items)
        };
        assert_eq!(
            search(SearchStrategy::LevelBeam),
            search(SearchStrategy::BestFirst)
        );
    }
}

/// Query (4, 4), search_exp=2 on the 3-level tree. Level 0 pops both root entries, queuing
/// lvl_2 nodes 0 (distance 32), 1 (18), 2 (72) and 3 (98). Level 1 pops nodes 1 and 0,
/// level 2 pops leaves 1 and 0, so `items` gets items 1 and 0 and nodes 2 and 3 stay queued.
#[test]
fn level_beam_descent_pops_search_exp_nodes_per_level() {
    let index = build_three_level_test_index();
    let query: Array1<f32> = array![4.0, 4.0];

    let (items, query_id) = index
        .new_search(query, 1, 2, 0, &HashSet::new(), SearchStrategy::LevelBeam)
        .unwrap();

    assert_eq!(ids(&items), vec![1]);
    assert_eq!(
        level_heap_node_ids(&index, query_id),
        vec![vec![], vec![2, 3], vec![]]
    );
}

/// Query (1, 1), search_exp=1 on the 2-level tree. new_search pops root entry 1, queues
/// leaves 1, 2 and 3, and scans leaf 1, so `items` holds item 3 after returning item 2.
/// The resume finds the leaf heap holding search_exp nodes, so it scans leaf 2 (items 4,
/// 5) even though root entry 0 is closer.
#[test]
fn level_beam_resumes_from_the_deepest_level_holding_search_exp() {
    let index = build_test_index();
    let query: Array1<f32> = array![1.0, 1.0];
    let (first, query_id) = index
        .new_search(query, 1, 1, 0, &HashSet::new(), SearchStrategy::LevelBeam)
        .unwrap();
    assert_eq!(ids(&first), vec![2]);

    let next = index
        .get_next_k_items(query_id, 3, 1, 0, &HashSet::new())
        .unwrap();

    assert_eq!(ids(&next), vec![3, 4, 5]);
    assert_eq!(
        level_heap_node_ids(&index, query_id),
        vec![vec![0], vec![3]]
    );
}

/// As above, but the resume uses search_exp=3. No level holds 3 (root has entry 0, the leaf
/// heap leaves 2 and 3), so the descent pops root entry 0, queues leaf 0 and scans
/// all three leaves. Items 3 and 1 are then the closest.
#[test]
fn level_beam_falls_back_to_the_root_when_no_level_holds_search_exp() {
    let index = build_test_index();
    let query: Array1<f32> = array![1.0, 1.0];
    let (_, query_id) = index
        .new_search(query, 1, 1, 0, &HashSet::new(), SearchStrategy::LevelBeam)
        .unwrap();

    let next = index
        .get_next_k_items(query_id, 2, 3, 0, &HashSet::new())
        .unwrap();

    assert_eq!(ids(&next), vec![3, 1]);
    assert_eq!(
        level_heap_node_ids(&index, query_id),
        vec![Vec::<u32>::new(), Vec::new()]
    );
}

/// Query (0, 0), k=4, search_exp=1 on the 3-level tree. The first descent finds item 0.
/// search_exp doubles to 2 and, with no level holding 2, the next descent starts at the
/// root and finds items 1 and 2. max_increments=1 stops there; with -1, search_exp doubles
/// to 4 and a descent from the now empty root heap scans lvl_2 node 3 and finds item 3.
#[test]
fn level_beam_doubles_search_exp_until_max_increments() {
    let search = |max_increments| {
        let index = build_three_level_test_index();
        let query: Array1<f32> = array![0.0, 0.0];
        let (items, _) = index
            .new_search(
                query,
                4,
                1,
                max_increments,
                &HashSet::new(),
                SearchStrategy::LevelBeam,
            )
            .unwrap();
        ids(&items)
    };

    assert_eq!(search(1), vec![0, 1, 2]);
    assert_eq!(search(-1), vec![0, 1, 2, 3]);
}

/// In a 1-level index, root's entries are leaves, so a descent with search_exp=2 scans
/// leaves 0 and 1 and leaves 2 and 3 queued.
#[test]
fn level_beam_on_a_1_level_index_pops_search_exp_leaves_from_the_root() {
    let index = build_ivf_style_index();
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, query_id) = index
        .new_search(query, 1, 2, 0, &HashSet::new(), SearchStrategy::LevelBeam)
        .unwrap();

    assert_eq!(ids(&items), vec![0]);
    assert_eq!(level_heap_node_ids(&index, query_id), vec![vec![2, 3]]);
}

/// A 2-level index whose `lvl_1/node_1` was never written. Query (4, 4) with search_exp=1
/// pops root entry 1 at (5, 5) first and skips it, so root entry 0 still gets the slot
/// and its leaf 1 yields item 1.
#[test]
fn level_beam_skips_an_unwritten_node_without_using_up_search_exp() {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", false);
    write_item_counts(&store, 2);
    write_index_root(&store, &array![[0.0f32, 0.0], [5.0, 5.0]]);
    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0], [1.0, 1.0]],
        "node_ids",
        &array![0u32, 1],
    );
    write_node(
        &store,
        "/lvl_2/node_0",
        &array![[0.0f32, 0.0]],
        "item_ids",
        &array![0u32],
    );
    write_node(
        &store,
        "/lvl_2/node_1",
        &array![[1.0f32, 1.0]],
        "item_ids",
        &array![1u32],
    );
    let index = Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let query: Array1<f32> = array![4.0, 4.0];

    let (items, _) = index
        .new_search(query, 1, 1, 0, &HashSet::new(), SearchStrategy::LevelBeam)
        .unwrap();

    assert_eq!(ids(&items), vec![1]);
}

/// search_exp=0 is rejected under both strategies, by every call that searches.
#[test]
fn search_exp_of_0_is_invalid_input() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];
    for strategy in [SearchStrategy::BestFirst, SearchStrategy::LevelBeam] {
        let result = index.new_search(query.clone(), 1, 0, -1, &HashSet::new(), strategy);
        assert!(matches!(result, Err(EcpError::InvalidInput(_))));
    }

    let (_, query_id) = index
        .new_search(query, 1, 1, -1, &HashSet::new(), SearchStrategy::BestFirst)
        .unwrap();
    assert!(matches!(
        index.get_next_k_items(query_id, 1, 0, -1, &HashSet::new()),
        Err(EcpError::InvalidInput(_))
    ));
    assert!(matches!(
        index.incremental_search(query_id, 1, 0, -1),
        Err(EcpError::InvalidInput(_))
    ));
}
