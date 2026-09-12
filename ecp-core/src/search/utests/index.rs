use super::*;
use crate::test_fixtures::{
    as_readable_listable, as_readable_writable_listable, new_memory_store, write_index_info,
    write_index_root, write_node, write_rep_item_ids, write_total_items,
};
use ndarray::array;
use std::sync::Barrier;
use std::thread;

/// Builds a node cache pre-populated with `((level, node_id), Node)`
/// entries, for fixtures that hand-construct an `Index` via struct literal
/// instead of `Index::load`.
fn nodes_cache(
    entries: impl IntoIterator<Item = ((usize, u32), Node)>,
) -> Cache<(usize, u32), Arc<Node>> {
    let cache = Cache::builder().build();
    for (key, node) in entries {
        cache.insert(key, Arc::new(node));
    }
    cache
}

#[test]
fn load_from_store_reconstructs_ivf_style_index_and_searches_correctly() {
    let store = new_memory_store();
    write_index_info(&store, 1, "L2", false);
    write_total_items(&store, 8);
    write_index_root(
        &store,
        &array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]],
    );

    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0], [0.4, 0.4]],
        "item_ids",
        &array![0u32, 1],
    );
    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[1.0f32, 1.0], [1.4, 1.4]],
        "item_ids",
        &array![2u32, 3],
    );
    write_node(
        &store,
        "/lvl_1/node_2",
        &array![[10.0f32, 10.0], [10.4, 10.4]],
        "item_ids",
        &array![4u32, 5],
    );
    write_node(
        &store,
        "/lvl_1/node_3",
        &array![[11.0f32, 11.0], [11.4, 11.4]],
        "item_ids",
        &array![6u32, 7],
    );

    let index = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

#[test]
fn index_info_reads_all_5_fields_without_loading_the_tree() {
    let store = new_memory_store();
    write_index_info(&store, 2, "IP", true);
    write_total_items(&store, 1_000);
    write_rep_item_ids(&store, &array![0u32, 5, 10, 15]);

    let info = IndexInfo::load_from_store(as_readable_listable(&store));

    assert_eq!(info.levels, 2);
    assert_eq!(info.metric, Metric::IP);
    assert!(info.is_normalized);
    assert_eq!(info.total_items, 1_000);
    assert_eq!(info.total_representatives, 4);
}

/// `node_1` is written before `node_0` on purpose - if `load_from_store`
/// ordered nodes by whatever order the store happens to list them in rather
/// than parsing each `node_N` path's own numeric suffix, this would surface
/// it as search returning items in the wrong order.
#[test]
fn load_from_store_sorts_node_paths_by_numeric_suffix_regardless_of_write_order() {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", false);
    write_total_items(&store, 8);
    write_index_root(&store, &array![[0.0f32, 0.0], [1.0, 1.0]]);

    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[1.0f32, 1.0], [10.0, 10.0], [11.0, 11.0]],
        "node_ids",
        &array![1u32, 2, 3],
    );
    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0]],
        "node_ids",
        &array![0u32],
    );

    write_node(
        &store,
        "/lvl_2/node_0",
        &array![[0.0f32, 0.0], [0.4, 0.4]],
        "item_ids",
        &array![0u32, 1],
    );
    write_node(
        &store,
        "/lvl_2/node_1",
        &array![[1.0f32, 1.0], [1.4, 1.4]],
        "item_ids",
        &array![2u32, 3],
    );
    write_node(
        &store,
        "/lvl_2/node_2",
        &array![[10.0f32, 10.0], [10.4, 10.4]],
        "item_ids",
        &array![4u32, 5],
    );
    write_node(
        &store,
        "/lvl_2/node_3",
        &array![[11.0f32, 11.0], [11.4, 11.4]],
        "item_ids",
        &array![6u32, 7],
    );

    let index = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// A small 2-level eCP tree, sized and derived the way `ECPBuilder` would for
/// 8 items with target_cluster_items=2 and levels=2:
///   total_clusters = ceil(N / target_cluster_items) = ceil(8 / 2) = 4 leaders
///   node_size      = ceil(total_clusters ** (1/levels)) = ceil(sqrt(4)) = 2
///
/// Leaders are picked by striding, exactly like `select_cluster_representatives`'s
/// default "offset" option (`representative_ids = item_ids[::target_cluster_items]`):
/// every 2nd item id, i.e. item ids 0, 2, 4, 6. A leader's *global id* (0..3, its
/// position in that strided list) is what lvl_1's "node_ids" and lvl_2's group
/// number refer to - it is not the same number as the item id it was struck from.
///
///   items (id=vector):  0=(0,0)  1=(0.4,0.4)  2=(1,1)  3=(1.4,1.4)
///                       4=(10,10) 5=(10.4,10.4) 6=(11,11) 7=(11.4,11.4)
///   leaders (global id = item id):  0=item 0   1=item 2   2=item 4   3=item 6
///   root:  first node_size (2) leaders -> [leader 0, leader 1]
///
/// lvl_1 buckets every leader under its nearest root leader (striding doesn't
/// promise a balanced split, so this one naturally comes out 1-vs-3):
///   lvl_1/node_0 = {leader 0}            (root leader 0 is its own only member)
///   lvl_1/node_1 = {leader 1, 2, 3}
///
/// lvl_2 (leaf) buckets every real item under its nearest leader overall, which
/// does come out even here (2 items per leader, matching target_cluster_items):
///   lvl_2/node_0 = {items 0,1}   lvl_2/node_1 = {items 2,3}
///   lvl_2/node_2 = {items 4,5}   lvl_2/node_3 = {items 6,7}
///
/// Nearest-to-farthest from the origin query (0,0) is therefore item id order:
/// 0, 1, 2, 3, 4, 5, 6, 7.
fn build_test_index(metric: Metric) -> Index {
    let store = new_memory_store();
    write_index_root(&store, &array![[0.0f32, 0.0], [1.0, 1.0]]);

    // lvl_1: node_ids are the *global leader ids* assigned to that root leader.
    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0]], // leader 0 (item 0)
        "node_ids",
        &array![0u32],
    );
    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[1.0f32, 1.0], [10.0, 10.0], [11.0, 11.0]], // leaders 1, 2, 3 (items 2, 4, 6)
        "node_ids",
        &array![1u32, 2, 3],
    );

    // lvl_2: leaf level, so children are "item_ids" (dataset ids) instead.
    // Group numbers 0/1/2/3 line up with leader global ids 0/1/2/3 above.
    write_node(
        &store,
        "/lvl_2/node_0",
        &array![[0.0f32, 0.0], [0.4, 0.4]],
        "item_ids",
        &array![0u32, 1],
    );
    write_node(
        &store,
        "/lvl_2/node_1",
        &array![[1.0f32, 1.0], [1.4, 1.4]],
        "item_ids",
        &array![2u32, 3],
    );
    write_node(
        &store,
        "/lvl_2/node_2",
        &array![[10.0f32, 10.0], [10.4, 10.4]],
        "item_ids",
        &array![4u32, 5],
    );
    write_node(
        &store,
        "/lvl_2/node_3",
        &array![[11.0f32, 11.0], [11.4, 11.4]],
        "item_ids",
        &array![6u32, 7],
    );

    let nodes = nodes_cache([
        (
            (0, 0),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_0".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (0, 1),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_1".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (1, 0),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_0".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (1, 1),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_1".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (1, 2),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_2".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (1, 3),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_3".to_string(),
                "item_ids".to_string(),
            ),
        ),
    ]);

    // Struct literal, not `Index::load`, so the fixture can use an
    // in-memory store instead of a real `FilesystemStore`.
    Index {
        store: as_readable_writable_listable(&store),
        metric,
        is_normalized: false,
        levels: 2,
        root: array![[0.0f32, 0.0], [1.0, 1.0]], // leader 0 (item 0), leader 1 (item 2)
        nodes,
        queries: Cache::builder().build(),
        next_query_id: AtomicUsize::new(0),
        memory_limit_bytes: None,
        accepting: AtomicBool::new(true),
        leaf_locks: DashMap::new(),
        total_items: Mutex::new(8),
    }
}

#[test]
fn l2_search_returns_nearest_items_in_order() {
    let index = build_test_index(Metric::L2);
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
    let index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 8, 100, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_tight_memory_limit_evicts_but_still_searches_correctly() {
    let mut index = build_test_index(Metric::L2);
    // Smaller than the ~144 bytes all 6 nodes would take resident at once,
    // but at least as large as the biggest single node (36 bytes), so the
    // node just touched is never itself evicted.
    index.set_memory_limit_bytes(Some(40));
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "eviction must not change search results"
    );
    index.nodes.run_pending_tasks();
    assert!(
        index.resident_bytes() <= 40,
        "resident bytes ({}) exceeded the limit",
        index.resident_bytes()
    );

    let still_present = index.nodes.entry_count();
    assert!(
        still_present < 6,
        "expected eviction to have freed at least one of the 6 touched nodes"
    );
}

#[test]
fn set_memory_limit_bytes_evicts_immediately_if_already_over_the_new_limit() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];
    index.new_search(query, 4, 4, -1, &HashSet::new());
    index.nodes.run_pending_tasks();
    assert_eq!(
        index.nodes.entry_count(),
        6,
        "sanity check: all 6 nodes loaded with no limit set"
    );

    index.set_memory_limit_bytes(Some(40));

    assert!(
        index.resident_bytes() <= 40,
        "resident bytes ({}) exceeded the limit right after lowering it",
        index.resident_bytes()
    );
    let still_present = index.nodes.entry_count();
    assert!(
        still_present < 6,
        "lowering the limit below current usage must evict immediately, not lazily"
    );
}

#[test]
fn l2_search_respects_exclude_set() {
    let index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];
    let exclude: HashSet<u32> = [0].into_iter().collect();

    let (items, _query_id) = index.new_search(query, 4, 4, -1, &exclude);

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![1, 2, 3, 4], "excluded item 0 must not appear");
}

#[test]
fn incremental_search_resumes_and_drains_remaining_items() {
    let index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    // First page: nearest 2 items.
    let (first, query_id) = index.new_search(query, 2, 4, -1, &HashSet::new());
    assert_eq!(
        first.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![0, 1]
    );

    // Second page: continues from where the first left off, same query_id.
    let second = index.get_next_k_items(query_id, 2, 4, -1, &HashSet::new());
    assert_eq!(
        second.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

/// k=1, search_exp=1 explores only the nearest leaf (items 0 and 1, both
/// pushed since a leaf is never partially processed) and drains 1, leaving
/// item 1 buffered.
#[test]
fn get_next_k_items_tops_up_a_partially_filled_buffer_below_k() {
    let index = build_test_index(Metric::L2);
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
/// ids 0,1) - not enough for k=4. With max_increments=-1 (unlimited), the
/// retry path at index.rs's `leaf_cnt == search_exp` check must double
/// search_exp (1 -> 2) and keep going, exploring a 2nd cluster (ids 2,3) to
/// reach k. Never exercised before: every existing fixture used a search_exp
/// large enough to satisfy k on the first pass.
#[test]
fn search_exp_doubles_until_k_items_found_with_unlimited_retries() {
    let index = build_test_index(Metric::L2);
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
/// max_increments=1 - i.e. "at most 1 retry" - which should be just enough
/// to go from search_exp=1 to 2 and reach k=4, identically to the unlimited
/// case. This isolates the finite-counter comparison (`increments >
/// max_increments`) from the `max_increments == -1` special case, which is
/// the only branch the test above exercises.
#[test]
fn finite_max_increments_still_allows_configured_number_of_retries() {
    let index = build_test_index(Metric::L2);
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
    let index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 8, 1, 1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "should stop after its 1 permitted retry (2 clusters explored), not find all 8 items"
    );
}

/// `Index.queries: Vec<QueryState>` holds every in-flight query, keyed by the
/// `query_id` returned from `new_search`. Every other test so far only ever
/// runs one query at a time, so cross-query indexing bugs (e.g. a query's
/// `tree_pq`/`items` bleeding into another's) would go unnoticed. This test
/// opens two queries against the *same* `Index` from opposite corners of the
/// fixture - query A from the origin (nearest-to-farthest: 0,1,2,3,4,5,6,7),
/// query B from (11,11), exactly item 6's position (nearest-to-farthest:
/// 6,7,5,4,3,2,1,0) - and interleaves `get_next_k_items` calls on both
/// `query_id`s, asserting each stream stays independent throughout.
#[test]
fn interleaved_queries_on_the_same_index_stay_independent() {
    let index = build_test_index(Metric::L2);
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

/// A levels=1 index is IVF-style: since node_size = ceil(total_clusters**1) =
/// total_clusters, the root holds *every* leader directly, and the single
/// level (nodes[0]) holds the leaf clusters - there is no intermediate level
/// to descend through. Same 8 items/4 leaders as `build_test_index`, just
/// flattened: root = all 4 leaders, and each leader's cluster is looked up
/// directly by its global id.
fn build_ivf_style_index(metric: Metric) -> Index {
    let store = new_memory_store();
    write_index_root(
        &store,
        &array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]],
    );

    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0], [0.4, 0.4]],
        "item_ids",
        &array![0u32, 1],
    );
    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[1.0f32, 1.0], [1.4, 1.4]],
        "item_ids",
        &array![2u32, 3],
    );
    write_node(
        &store,
        "/lvl_1/node_2",
        &array![[10.0f32, 10.0], [10.4, 10.4]],
        "item_ids",
        &array![4u32, 5],
    );
    write_node(
        &store,
        "/lvl_1/node_3",
        &array![[11.0f32, 11.0], [11.4, 11.4]],
        "item_ids",
        &array![6u32, 7],
    );

    let nodes = nodes_cache([
        (
            (0, 0),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_0".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (0, 1),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_1".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (0, 2),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_2".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (0, 3),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_3".to_string(),
                "item_ids".to_string(),
            ),
        ),
    ]);

    Index {
        store: as_readable_writable_listable(&store),
        metric,
        is_normalized: false,
        levels: 1,
        root: array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]], // all 4 leaders
        nodes,
        queries: Cache::builder().build(),
        next_query_id: AtomicUsize::new(0),
        memory_limit_bytes: None,
        accepting: AtomicBool::new(true),
        leaf_locks: DashMap::new(),
        total_items: Mutex::new(8),
    }
}

/// A 3-level tree: root -> lvl_1 -> lvl_2 -> lvl_3 (leaf). Every fixture so
/// far tops out at levels=2, where the intermediate level (nodes[0]) always
/// has `(level + 1) == (levels - 1)`, so its children are pushed straight as
/// leaves (index.rs's `if` branch of that check). With levels=3, lvl_1's
/// children (into lvl_2) instead take the `else` branch - pushed as another
/// non-leaf level - which no existing test reaches. Only lvl_2's children
/// (into lvl_3) hit the leaf branch.
///
/// Kept deliberately linear (one child per intermediate node) so the descent
/// path is unambiguous: root has 2 leaders, each lvl_1 node fans out to 2
/// lvl_2 nodes, and each lvl_2 node has exactly 1 lvl_3 child - 4 items total,
/// laid out on a line so nearest-to-farthest from the origin query is item
/// id order, same shape of assertion as `l2_search_returns_nearest_items_in_order`.
fn build_three_level_test_index() -> Index {
    let store = new_memory_store();
    write_index_root(&store, &array![[0.0f32, 0.0], [10.0, 10.0]]);

    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0], [1.0, 1.0]],
        "node_ids",
        &array![0u32, 1],
    );
    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[10.0f32, 10.0], [11.0, 11.0]],
        "node_ids",
        &array![2u32, 3],
    );

    write_node(
        &store,
        "/lvl_2/node_0",
        &array![[0.0f32, 0.0]],
        "node_ids",
        &array![0u32],
    );
    write_node(
        &store,
        "/lvl_2/node_1",
        &array![[1.0f32, 1.0]],
        "node_ids",
        &array![1u32],
    );
    write_node(
        &store,
        "/lvl_2/node_2",
        &array![[10.0f32, 10.0]],
        "node_ids",
        &array![2u32],
    );
    write_node(
        &store,
        "/lvl_2/node_3",
        &array![[11.0f32, 11.0]],
        "node_ids",
        &array![3u32],
    );

    write_node(
        &store,
        "/lvl_3/node_0",
        &array![[0.0f32, 0.0]],
        "item_ids",
        &array![0u32],
    );
    write_node(
        &store,
        "/lvl_3/node_1",
        &array![[1.0f32, 1.0]],
        "item_ids",
        &array![1u32],
    );
    write_node(
        &store,
        "/lvl_3/node_2",
        &array![[10.0f32, 10.0]],
        "item_ids",
        &array![2u32],
    );
    write_node(
        &store,
        "/lvl_3/node_3",
        &array![[11.0f32, 11.0]],
        "item_ids",
        &array![3u32],
    );

    let nodes = nodes_cache([
        (
            (0, 0),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_0".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (0, 1),
            Node::new(
                as_readable_listable(&store),
                "/lvl_1/node_1".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (1, 0),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_0".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (1, 1),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_1".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (1, 2),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_2".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (1, 3),
            Node::new(
                as_readable_listable(&store),
                "/lvl_2/node_3".to_string(),
                "node_ids".to_string(),
            ),
        ),
        (
            (2, 0),
            Node::new(
                as_readable_listable(&store),
                "/lvl_3/node_0".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (2, 1),
            Node::new(
                as_readable_listable(&store),
                "/lvl_3/node_1".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (2, 2),
            Node::new(
                as_readable_listable(&store),
                "/lvl_3/node_2".to_string(),
                "item_ids".to_string(),
            ),
        ),
        (
            (2, 3),
            Node::new(
                as_readable_listable(&store),
                "/lvl_3/node_3".to_string(),
                "item_ids".to_string(),
            ),
        ),
    ]);

    Index {
        store: as_readable_writable_listable(&store),
        metric: Metric::L2,
        is_normalized: false,
        levels: 3,
        root: array![[0.0f32, 0.0], [10.0, 10.0]],
        nodes,
        queries: Cache::builder().build(),
        next_query_id: AtomicUsize::new(0),
        memory_limit_bytes: None,
        accepting: AtomicBool::new(true),
        leaf_locks: DashMap::new(),
        total_items: Mutex::new(4),
    }
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
    let index = build_ivf_style_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    // In a levels=1 tree every popped node is a leaf, so leaf_cnt (what
    // search_exp actually counts) advances once per cluster regardless of
    // how many total node lookups that involves - search_exp=4 here means
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
    let index = build_test_index(Metric::L2);

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
    let index = Arc::new(build_test_index(Metric::L2));
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

/// Builds the same 4-leader/8-item IVF-style fixture used by
/// `load_from_store_reconstructs_ivf_style_index_and_searches_correctly`,
/// but hands back the backing store too, so a test can load a second,
/// independent `Index` against it later to simulate a process restart.
fn write_ivf_style_fixture() -> Arc<zarrs::storage::store::MemoryStore> {
    let store = new_memory_store();
    write_index_info(&store, 1, "L2", false);
    write_total_items(&store, 8);
    write_index_root(
        &store,
        &array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]],
    );
    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0], [0.4, 0.4]],
        "item_ids",
        &array![0u32, 1],
    );
    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[1.0f32, 1.0], [1.4, 1.4]],
        "item_ids",
        &array![2u32, 3],
    );
    write_node(
        &store,
        "/lvl_1/node_2",
        &array![[10.0f32, 10.0], [10.4, 10.4]],
        "item_ids",
        &array![4u32, 5],
    );
    write_node(
        &store,
        "/lvl_1/node_3",
        &array![[11.0f32, 11.0], [11.4, 11.4]],
        "item_ids",
        &array![6u32, 7],
    );
    store
}

#[test]
fn shutdown_then_reload_resumes_a_query_from_a_fresh_index() {
    let store = write_ivf_style_fixture();

    let first_process = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (first, query_id) = first_process.new_search(query, 2, 4, -1, &HashSet::new());
    assert_eq!(
        first.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    first_process.shutdown();

    let second_process = Index::load_from_store(as_readable_writable_listable(&store), None);
    let second = second_process.get_next_k_items(query_id, 2, 4, -1, &HashSet::new());
    assert_eq!(
        second.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn next_query_id_is_seeded_above_every_persisted_id_after_reload() {
    let store = write_ivf_style_fixture();

    let first_process = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = first_process.new_search(query.clone(), 2, 4, -1, &HashSet::new());
    first_process.shutdown();

    let second_process = Index::load_from_store(as_readable_writable_listable(&store), None);
    let (_, new_id) = second_process.new_search(query, 2, 4, -1, &HashSet::new());
    assert!(
        new_id > query_id,
        "new_id ({new_id}) must exceed the reloaded id ({query_id})"
    );
}

#[test]
fn shutdown_blocks_subsequent_calls_on_the_same_instance() {
    let index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = index.new_search(query.clone(), 2, 4, -1, &HashSet::new());
    index.shutdown();

    let after_shutdown = index.get_next_k_items(query_id, 2, 4, -1, &HashSet::new());
    assert!(
        after_shutdown.is_empty(),
        "get_next_k_items must no-op after shutdown"
    );

    let (new_items, new_id) = index.new_search(query, 2, 4, -1, &HashSet::new());
    assert!(new_items.is_empty(), "new_search must no-op after shutdown");
    assert_ne!(new_id, query_id, "a fresh id is still allocated");

    let never_found = index.get_next_k_items(new_id, 2, 4, -1, &HashSet::new());
    assert!(
        never_found.is_empty(),
        "the post-shutdown id was never actually searched or cached"
    );
}

#[test]
fn shutdown_is_idempotent() {
    let index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];
    index.new_search(query, 2, 4, -1, &HashSet::new());
    index.shutdown();
    index.shutdown();
}

/// Excluding every item forces `items` to stay empty even once `tree_pq` is
/// genuinely drained (not just not-yet-populated), the only way to reach a
/// real "nothing left to explore, nothing left to hand back" state.
#[test]
fn exhausted_query_is_erased_not_persisted() {
    let store = write_ivf_style_fixture();

    let index = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let exclude: HashSet<u32> = (0..8).collect();
    let (items, query_id) = index.new_search(query, 4, 4, -1, &exclude);
    assert!(
        items.is_empty(),
        "sanity check: excluding every item must leave nothing found"
    );
    index.shutdown();

    let reloaded = Index::load_from_store(as_readable_writable_listable(&store), None);
    let resumed = reloaded.get_next_k_items(query_id, 4, 4, -1, &HashSet::new());
    assert!(
        resumed.is_empty(),
        "an exhausted query must have been erased, not left resumable"
    );
}

#[test]
fn evicted_query_resumes_correctly_within_the_same_process() {
    let store = write_ivf_style_fixture();

    let mut index = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (first, query_id) = index.new_search(query, 1, 1, -1, &HashSet::new());
    assert_eq!(first.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![0]);

    // Capacity this tiny is below any real QueryState's weight, so
    // set_memory_limit_bytes's synchronous eviction pass evicts the query
    // just created.
    index.set_memory_limit_bytes(Some(20));
    assert_eq!(
        index.queries.entry_count(),
        0,
        "sanity check: the query must have actually been evicted"
    );

    let second = index.get_next_k_items(query_id, 1, 1, -1, &HashSet::new());
    assert_eq!(
        second.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![1],
        "an evicted query must resume from disk, not come back empty"
    );
}

/// Races `new_search` threads against a concurrent `shutdown()`. A query
/// still mid-publish when `shutdown` runs may be missing, that's expected;
/// only checks for no panic/deadlock and that whatever did persist is
/// internally coherent, no torn write.
#[test]
fn shutdown_racing_concurrent_searches_leaves_persisted_state_coherent() {
    let index = Arc::new(build_test_index(Metric::L2));
    const THREADS: usize = 8;
    const SEARCHES_PER_THREAD: usize = 20;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let index = Arc::clone(&index);
            std::thread::spawn(move || {
                for _ in 0..SEARCHES_PER_THREAD {
                    let query: Array1<f32> = array![0.0, 0.0];
                    let _ = index.new_search(query, 4, 4, -1, &HashSet::new());
                }
            })
        })
        .collect();

    index.shutdown();

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }

    for query_id in 0..(THREADS * SEARCHES_PER_THREAD) {
        if let Some(state) = persistence::load_query(&index.store, query_id) {
            let scores: Vec<f32> = state.items.iter().map(|(s, _)| s.into_inner()).collect();
            assert!(
                scores.windows(2).all(|w| w[0] <= w[1]),
                "persisted items for query_id={query_id} are not sorted: {scores:?}"
            );
        }
    }
}

#[test]
fn cleanup_persisted_queries_older_than_erases_stale_entries() {
    let store = write_ivf_style_fixture();
    let index = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = index.new_search(query, 1, 1, -1, &HashSet::new());
    index.shutdown();

    assert_eq!(
        index.cleanup_persisted_queries_older_than(0),
        0,
        "nothing is older than the Unix epoch"
    );

    let erased = index.cleanup_persisted_queries_older_than(u64::MAX);
    assert_eq!(erased, 1, "any real timestamp is older than u64::MAX");

    let reloaded = Index::load_from_store(as_readable_writable_listable(&store), None);
    let resumed = reloaded.get_next_k_items(query_id, 1, 1, -1, &HashSet::new());
    assert!(
        resumed.is_empty(),
        "cleanup must have erased the query, nothing left to resume"
    );
}

#[test]
fn repersisting_after_more_progress_reflects_the_latest_state() {
    let store = write_ivf_style_fixture();

    let first_process = Index::load_from_store(as_readable_writable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = first_process.new_search(query.clone(), 1, 1, -1, &HashSet::new());
    first_process.shutdown();

    let second_process = Index::load_from_store(as_readable_writable_listable(&store), None);
    let drained_more = second_process.get_next_k_items(query_id, 2, 1, -1, &HashSet::new());
    assert_eq!(
        drained_more.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![1, 2]
    );
    second_process.shutdown();

    let third_process = Index::load_from_store(as_readable_writable_listable(&store), None);
    let remaining = third_process.get_next_k_items(query_id, 10, 1, -1, &HashSet::new());
    assert_eq!(
        remaining.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![3, 4, 5, 6, 7],
        "must continue from the second process's progress, not the first's stale state"
    );
}

#[test]
fn insert_invalidates_exactly_the_touched_cache_entry() {
    let index = build_test_index(Metric::L2);
    let before: Vec<((usize, u32), Arc<Node>)> =
        index.nodes.iter().map(|(key, node)| (*key, node)).collect();
    assert_eq!(before.len(), 6, "sanity check: every node is pre-cached");

    // Close enough to item 0 (in leaf (1, 0)) to route to that exact leaf.
    index.insert(array![[0.01f32, 0.01]]);

    assert!(
        index.nodes.get(&(1, 0)).is_none(),
        "the touched leaf's entry must be invalidated"
    );
    for (key, node) in &before {
        if *key == (1, 0) {
            continue;
        }
        let still_there = index
            .nodes
            .get(key)
            .unwrap_or_else(|| panic!("untouched entry {key:?} must not have been evicted"));
        assert!(
            Arc::ptr_eq(node, &still_there),
            "untouched entry {key:?} must keep its original identity"
        );
    }
}

/// Inserts into 4 disjoint leaves from 4 threads at once (behind a
/// `Barrier`, so they actually overlap). Correctness-focused: proves no
/// data race/corruption when leaves don't overlap, not the absence of
/// blocking (which isn't directly observable from a test).
#[test]
fn concurrent_inserts_to_different_leaves_do_not_block_each_other() {
    for _ in 0..20 {
        let index = build_test_index(Metric::L2);
        let barrier = Barrier::new(4);
        let new_points: [[f32; 2]; 4] =
            [[0.01, 0.01], [1.01, 1.01], [10.01, 10.01], [11.01, 11.01]];

        // Each thread's assigned id depends on reservation order, which is
        // non-deterministic, so capture what insert actually returns rather
        // than assuming a fixed id per point.
        let assigned: Vec<(u32, [f32; 2])> = thread::scope(|scope| {
            let handles: Vec<_> = new_points
                .into_iter()
                .map(|point| {
                    let index = &index;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        (index.insert(array![point]).start, point)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread panicked"))
                .collect()
        });

        for (id, point) in assigned {
            let (items, _) =
                index.new_search(Array1::from_vec(point.to_vec()), 1, 4, -1, &HashSet::new());
            assert_eq!(
                items.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                vec![id],
                "item {id} must be found after concurrent inserts to disjoint leaves"
            );
        }
    }
}

/// One thread repeatedly searches leaf (1, 0) while another concurrently
/// inserts several new points into that same leaf. Proves the leaf's read
/// lock (search) and write lock (insert) actually serialize correctly
/// instead of racing on the same on-disk arrays.
#[test]
fn concurrent_insert_and_search_on_the_same_leaf_do_not_corrupt_data() {
    for _ in 0..20 {
        let index = build_test_index(Metric::L2);
        // Marks every entry evicted, then forces that eviction to run now.
        // Which leads to node_at repopulating each leaf from scratch,
        // under its lock, instead of returning a pre-seeded node.
        index.nodes.invalidate_all();
        index.nodes.run_pending_tasks();
        let barrier = Barrier::new(2);

        let assigned_ids: Vec<u32> = thread::scope(|scope| {
            let searcher = &index;
            let searcher_barrier = &barrier;
            scope.spawn(move || {
                searcher_barrier.wait();
                for _ in 0..50 {
                    searcher.new_search(array![0.0f32, 0.0], 10, 4, -1, &HashSet::new());
                }
            });

            let inserter = &index;
            let inserter_barrier = &barrier;
            let handle = scope.spawn(move || {
                inserter_barrier.wait();
                (0..10u32)
                    .map(|i| {
                        inserter
                            .insert(array![[0.01f32 + i as f32 * 0.001, 0.01]])
                            .start
                    })
                    .collect::<Vec<u32>>()
            });
            handle.join().expect("inserter thread panicked")
        });

        let (items, _) = index.new_search(array![0.0f32, 0.0], 12, 4, -1, &HashSet::new());
        let mut ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        ids.sort_unstable();
        let mut expected: Vec<u32> = assigned_ids;
        expected.extend([0, 1]);
        expected.sort_unstable();
        assert_eq!(
            ids, expected,
            "every concurrently inserted item plus the original leaf contents must be present, with no duplicates or losses"
        );
    }
}
