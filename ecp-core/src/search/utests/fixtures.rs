use super::*;
use crate::test_fixtures::{
    as_readable_listable, as_readable_writable_listable, new_memory_store, write_index_info,
    write_index_root, write_item_counts, write_node,
};
use ndarray::array;

/// Builds a node cache pre-populated with `((level, node_id), Node)`
/// entries, for fixtures that hand-construct an `Index` via struct literal
/// instead of `Index::load`.
pub(super) fn nodes_cache(
    entries: impl IntoIterator<Item = ((usize, u32), Node)>,
) -> Cache<(usize, u32), Arc<Node>> {
    let cache = Cache::builder().build();
    for (key, node) in entries {
        cache.insert(key, Arc::new(node));
    }
    cache
}

/// A small 2-level eCP tree, sized and derived the way `Builder` would for
/// 8 items with target_cluster_items=2 and levels=2:
///   total_clusters = ceil(N / target_cluster_items) = ceil(8 / 2) = 4 leaders
///   node_size      = ceil(total_clusters ** (1/levels)) = ceil(sqrt(4)) = 2
///
/// Leaders are picked by striding, like `RepresentativeStrategy::Offset`
/// (`(0..total_items).step_by(target_cluster_items)`): every 2nd item id,
/// i.e. item ids 0, 2, 4, 6. A leader's *global id* (0..3, its
/// position in that strided list) is what lvl_1's "node_ids" and lvl_2's group
/// number refer to. It is not the same number as the item id it was struck from.
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
/// lvl_2 (leaf) buckets every item under its nearest leader overall, which
/// does come out even here (2 items per leader, matching target_cluster_items):
///   lvl_2/node_0 = {items 0,1}   lvl_2/node_1 = {items 2,3}
///   lvl_2/node_2 = {items 4,5}   lvl_2/node_3 = {items 6,7}
///
/// Nearest-to-farthest from the origin query (0,0) is therefore item id order:
/// 0, 1, 2, 3, 4, 5, 6, 7.
pub(super) fn build_test_index(metric: Metric) -> Index {
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
        next_item_id: Mutex::new(8),
        total_items: Mutex::new(8),
    }
}

/// A levels=1 index is IVF-style: since node_size = ceil(total_clusters**1) =
/// total_clusters, the root holds *every* leader directly, and the single
/// level (nodes[0]) holds the leaf clusters. There is no intermediate level
/// to descend through. Same 8 items/4 leaders as `build_test_index`, just
/// flattened: root = all 4 leaders, and each leader's cluster is looked up
/// directly by its global id.
pub(super) fn build_ivf_style_index(metric: Metric) -> Index {
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
        next_item_id: Mutex::new(8),
        total_items: Mutex::new(8),
    }
}

/// A 3-level tree: root -> lvl_1 -> lvl_2 -> lvl_3 (leaf). Every fixture so
/// far tops out at levels=2, where the intermediate level (nodes[0]) always
/// has `(level + 1) == (levels - 1)`, so its children are pushed straight as
/// leaves (query.rs's `if` branch of that check). With levels=3, lvl_1's
/// children (into lvl_2) instead take the `else` branch, pushed as another
/// non-leaf level, which no existing test reaches. Only lvl_2's children
/// (into lvl_3) hit the leaf branch.
///
/// Kept deliberately linear (one child per intermediate node) so the descent
/// path is unambiguous: root has 2 leaders, each lvl_1 node fans out to 2
/// lvl_2 nodes, and each lvl_2 node has exactly 1 lvl_3 child, 4 items total,
/// laid out on a line so nearest-to-farthest from the origin query is item
/// id order, same shape of assertion as `l2_search_returns_nearest_items_in_order`.
pub(super) fn build_three_level_test_index() -> Index {
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
        next_item_id: Mutex::new(4),
        total_items: Mutex::new(4),
    }
}

/// Builds the same 4-leader/8-item IVF-style fixture used by
/// `load_from_store_reconstructs_ivf_style_index_and_searches_correctly`,
/// but hands back the backing store too, so a test can load a second,
/// independent `Index` against it later to simulate a process restart.
pub(super) fn write_ivf_style_fixture() -> Arc<zarrs::storage::store::MemoryStore> {
    let store = new_memory_store();
    write_index_info(&store, 1, "L2", false);
    write_item_counts(&store, 8);
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
