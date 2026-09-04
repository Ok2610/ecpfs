use super::*;
use crate::test_fixtures::{
    as_readable_listable, new_memory_store, write_index_info, write_index_root, write_node,
};
use ndarray::array;

#[test]
fn load_from_store_reconstructs_ivf_style_index_and_searches_correctly() {
    let store = new_memory_store();
    write_index_info(&store, 1, "L2", false);
    write_index_root(&store, &array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]]);

    write_node(&store, "/lvl_1/node_0", &array![[0.0f32, 0.0], [0.4, 0.4]], "item_ids", &array![0u32, 1]);
    write_node(&store, "/lvl_1/node_1", &array![[1.0f32, 1.0], [1.4, 1.4]], "item_ids", &array![2u32, 3]);
    write_node(&store, "/lvl_1/node_2", &array![[10.0f32, 10.0], [10.4, 10.4]], "item_ids", &array![4u32, 5]);
    write_node(&store, "/lvl_1/node_3", &array![[11.0f32, 11.0], [11.4, 11.4]], "item_ids", &array![6u32, 7]);

    let mut index = Index::load_from_store(as_readable_listable(&store), None);
    let query: Array1<f32> = array![0.0, 0.0];
    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

/// `node_1` is written before `node_0` on purpose - if `load_from_store`
/// ordered nodes by whatever order the store happens to list them in rather
/// than parsing each `node_N` path's own numeric suffix, this would surface
/// it as search returning items in the wrong order.
#[test]
fn load_from_store_sorts_node_paths_by_numeric_suffix_regardless_of_write_order() {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", false);
    write_index_root(&store, &array![[0.0f32, 0.0], [1.0, 1.0]]);

    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[1.0f32, 1.0], [10.0, 10.0], [11.0, 11.0]],
        "node_ids",
        &array![1u32, 2, 3],
    );
    write_node(&store, "/lvl_1/node_0", &array![[0.0f32, 0.0]], "node_ids", &array![0u32]);

    write_node(&store, "/lvl_2/node_0", &array![[0.0f32, 0.0], [0.4, 0.4]], "item_ids", &array![0u32, 1]);
    write_node(&store, "/lvl_2/node_1", &array![[1.0f32, 1.0], [1.4, 1.4]], "item_ids", &array![2u32, 3]);
    write_node(&store, "/lvl_2/node_2", &array![[10.0f32, 10.0], [10.4, 10.4]], "item_ids", &array![4u32, 5]);
    write_node(&store, "/lvl_2/node_3", &array![[11.0f32, 11.0], [11.4, 11.4]], "item_ids", &array![6u32, 7]);

    let mut index = Index::load_from_store(as_readable_listable(&store), None);
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

    let lvl_1 = vec![
        Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "node_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_1/node_1".to_string(), "node_ids".to_string()),
    ];
    let lvl_2 = vec![
        Node::new(as_readable_listable(&store), "/lvl_2/node_0".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_2/node_1".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_2/node_2".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_2/node_3".to_string(), "item_ids".to_string()),
    ];

    // Struct literal, not `Index::load`, so the fixture can use an
    // in-memory store instead of a real `FilesystemStore`.
    Index {
        metric,
        is_normalized: false,
        levels: 2,
        root: array![[0.0f32, 0.0], [1.0, 1.0]], // leader 0 (item 0), leader 1 (item 2)
        nodes: vec![lvl_1, lvl_2],
        queries: Vec::new(),
        memory_limit_bytes: None,
        lru: LruCache::unbounded(),
        resident_bytes: 0,
    }
}

#[test]
fn l2_search_returns_nearest_items_in_order() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    // search_exp=4 explores all 4 leaf nodes, so this is an exact top-4.
    let (items, _query_id) =
        index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);

    let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
    assert!(scores.windows(2).all(|w| w[0] <= w[1]), "scores not sorted: {scores:?}");
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
    assert_eq!(ids, vec![0, 1, 2, 3], "eviction must not change search results");
    assert!(index.resident_bytes <= 40, "resident bytes ({}) exceeded the limit", index.resident_bytes);

    let still_loaded = index.nodes.iter().flatten().filter(|n| n.is_loaded()).count();
    assert!(still_loaded < 6, "expected eviction to have freed at least one of the 6 touched nodes");
}

#[test]
fn set_memory_limit_bytes_evicts_immediately_if_already_over_the_new_limit() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];
    index.new_search(query, 4, 4, -1, &HashSet::new());
    assert_eq!(index.nodes.iter().flatten().filter(|n| n.is_loaded()).count(), 6, "sanity check: all 6 nodes loaded with no limit set");

    index.set_memory_limit_bytes(Some(40));

    assert!(index.resident_bytes <= 40, "resident bytes ({}) exceeded the limit right after lowering it", index.resident_bytes);
    let still_loaded = index.nodes.iter().flatten().filter(|n| n.is_loaded()).count();
    assert!(still_loaded < 6, "lowering the limit below current usage must evict immediately, not lazily");
}

#[test]
fn l2_search_respects_exclude_set() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];
    let exclude: HashSet<u32> = [0].into_iter().collect();

    let (items, _query_id) =
        index.new_search(query, 4, 4, -1, &exclude);

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![1, 2, 3, 4], "excluded item 0 must not appear");
}

#[test]
fn incremental_search_resumes_and_drains_remaining_items() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    // First page: nearest 2 items.
    let (first, query_id) =
        index.new_search(query, 2, 4, -1, &HashSet::new());
    assert_eq!(first.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![0, 1]);

    // Second page: continues from where the first left off, same query_id.
    let second =
        index.get_next_k_items(query_id, 2, 4, -1, &HashSet::new());
    assert_eq!(second.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![2, 3]);
}

/// With search_exp=1, the first pass only explores 1 leaf cluster (2 items:
/// ids 0,1) - not enough for k=4. With max_increments=-1 (unlimited), the
/// retry path at index.rs's `leaf_cnt == search_exp` check must double
/// search_exp (1 -> 2) and keep going, exploring a 2nd cluster (ids 2,3) to
/// reach k. Never exercised before: every existing fixture used a search_exp
/// large enough to satisfy k on the first pass.
#[test]
fn search_exp_doubles_until_k_items_found_with_unlimited_retries() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) =
        index.new_search(query, 4, 1, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3], "doubling search_exp should eventually surface all 4 nearest items");
}

/// Same setup as the unlimited-retry test above, but with a *finite*
/// max_increments=1 - i.e. "at most 1 retry" - which should be just enough
/// to go from search_exp=1 to 2 and reach k=4, identically to the unlimited
/// case. This isolates the finite-counter comparison (`increments >
/// max_increments`) from the `max_increments == -1` special case, which is
/// the only branch the test above exercises.
#[test]
fn finite_max_increments_still_allows_configured_number_of_retries() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) =
        index.new_search(query, 4, 1, 1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "max_increments=1 should permit the single retry needed to reach k=4"
    );
}

/// Same fixture, but k=8 (all items) needs 2 doublings (search_exp 1 -> 2 -> 4)
/// to satisfy, while max_increments=1 permits only 1. The search must stop
/// once its retry budget is exhausted rather than looping until k is met -
/// returning fewer than k items (the 2 clusters/4 items reachable after the
/// single permitted retry), not all 8.
#[test]
fn search_stops_once_max_increments_is_exhausted() {
    let mut index = build_test_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) =
        index.new_search(query, 8, 1, 1, &HashSet::new());

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
    let mut index = build_test_index(Metric::L2);
    let query_a: Array1<f32> = array![0.0, 0.0];
    let query_b: Array1<f32> = array![11.0, 11.0];

    let (first_a, query_id_a) =
        index.new_search(query_a, 2, 4, -1, &HashSet::new());
    assert_eq!(first_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![0, 1]);

    let (first_b, query_id_b) =
        index.new_search(query_b, 2, 4, -1, &HashSet::new());
    assert_eq!(first_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![6, 7]);
    assert_ne!(query_id_a, query_id_b);

    // Interleaved: resume A, then B, then A again.
    let second_a =
        index.get_next_k_items(query_id_a, 2, 4, -1, &HashSet::new());
    assert_eq!(second_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![2, 3]);

    let second_b =
        index.get_next_k_items(query_id_b, 2, 4, -1, &HashSet::new());
    assert_eq!(second_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![5, 4]);

    let third_a =
        index.get_next_k_items(query_id_a, 2, 4, -1, &HashSet::new());
    assert_eq!(third_a.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![4, 5]);

    let third_b =
        index.get_next_k_items(query_id_b, 2, 4, -1, &HashSet::new());
    assert_eq!(third_b.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![3, 2]);
}

/// A levels=1 index is IVF-style: since node_size = ceil(total_clusters**1) =
/// total_clusters, the root holds *every* leader directly, and the single
/// level (nodes[0]) holds the leaf clusters - there is no intermediate level
/// to descend through. Same 8 items/4 leaders as `build_test_index`, just
/// flattened: root = all 4 leaders, and each leader's cluster is looked up
/// directly by its global id.
fn build_ivf_style_index(metric: Metric) -> Index {
    let store = new_memory_store();

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

    let leaf_clusters = vec![
        Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_1/node_1".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_1/node_2".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_1/node_3".to_string(), "item_ids".to_string()),
    ];

    Index {
        metric,
        is_normalized: false,
        levels: 1,
        root: array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]], // all 4 leaders
        nodes: vec![leaf_clusters],
        queries: Vec::new(),
        memory_limit_bytes: None,
        lru: LruCache::unbounded(),
        resident_bytes: 0,
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

    let lvl_1 = vec![
        Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "node_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_1/node_1".to_string(), "node_ids".to_string()),
    ];
    let lvl_2 = vec![
        Node::new(as_readable_listable(&store), "/lvl_2/node_0".to_string(), "node_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_2/node_1".to_string(), "node_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_2/node_2".to_string(), "node_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_2/node_3".to_string(), "node_ids".to_string()),
    ];
    let lvl_3 = vec![
        Node::new(as_readable_listable(&store), "/lvl_3/node_0".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_3/node_1".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_3/node_2".to_string(), "item_ids".to_string()),
        Node::new(as_readable_listable(&store), "/lvl_3/node_3".to_string(), "item_ids".to_string()),
    ];

    Index {
        metric: Metric::L2,
        is_normalized: false,
        levels: 3,
        root: array![[0.0f32, 0.0], [10.0, 10.0]],
        nodes: vec![lvl_1, lvl_2, lvl_3],
        queries: Vec::new(),
        memory_limit_bytes: None,
        lru: LruCache::unbounded(),
        resident_bytes: 0,
    }
}

#[test]
fn three_level_tree_descends_through_intermediate_level() {
    let mut index = build_three_level_test_index();
    let query: Array1<f32> = array![0.0, 0.0];

    // search_exp=4 explores all 4 leaf nodes, so this is an exact top-4.
    let (items, _query_id) =
        index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);

    let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
    assert!(scores.windows(2).all(|w| w[0] <= w[1]), "scores not sorted: {scores:?}");
}

#[test]
fn levels_1_index_searches_like_ivf_without_panicking() {
    let mut index = build_ivf_style_index(Metric::L2);
    let query: Array1<f32> = array![0.0, 0.0];

    // In a levels=1 tree every popped node is a leaf, so leaf_cnt (what
    // search_exp actually counts) advances once per cluster regardless of
    // how many total node lookups that involves - search_exp=4 here means
    // "don't stop before all 4 clusters have been scanned", not "check 4
    // nodes" in general (those only coincide because there's nothing but
    // leaves in this particular tree).
    let (items, _query_id) =
        index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}
