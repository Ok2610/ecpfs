use super::*;
use crate::test_fixtures::{
    as_readable_writable_listable, new_memory_store, write_index_info, write_index_root,
    write_item_counts, write_node,
};
use ndarray::array;

/// Builds a small 2-level eCP tree, sized and derived the way `Builder`
/// would for 8 items with target_cluster_items=2 and levels=2:
///   total_clusters = ceil(N / target_cluster_items) = ceil(8 / 2) = 4 representatives
///   node_size      = ceil(total_clusters ** (1/levels)) = ceil(sqrt(4)) = 2
///
/// Representatives are picked by striding, like `RepresentativeStrategy::Offset`
/// (`(0..total_items).step_by(target_cluster_items)`), which picks every 2nd item
/// id: 0, 2, 4, 6. A representative's *global id* (0..3, its position in that
/// strided list) is what lvl_1's "node_ids" and lvl_2's group number refer to.
/// It is not the same number as the item id it was struck from.
///
///   items (id=vector):  0=(0,0)  1=(0.4,0.4)  2=(1,1)  3=(1.4,1.4)
///                       4=(10,10) 5=(10.4,10.4) 6=(11,11) 7=(11.4,11.4)
///   representatives (global id = item id):  0=item 0   1=item 2   2=item 4   3=item 6
///   root:  first node_size (2) representatives -> [representative 0, representative 1]
///
/// lvl_1 buckets every representative under its nearest root representative
/// (striding doesn't promise a balanced split, so this one comes out 1-vs-3):
///   lvl_1/node_0 = {representative 0}   (root representative 0 is its own only member)
///   lvl_1/node_1 = {representative 1, 2, 3}
///
/// lvl_2 (leaf) buckets every item under its nearest representative overall,
/// which does come out even here (2 items per representative, matching
/// target_cluster_items):
///   lvl_2/node_0 = {items 0,1}   lvl_2/node_1 = {items 2,3}
///   lvl_2/node_2 = {items 4,5}   lvl_2/node_3 = {items 6,7}
///
/// Nearest-to-farthest from the origin query (0,0) is therefore item id order:
/// 0, 1, 2, 3, 4, 5, 6, 7.
pub(super) fn build_test_index() -> Index {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", false);
    write_item_counts(&store, 8);
    // representative 0 (item 0), representative 1 (item 2)
    write_index_root(&store, &array![[0.0f32, 0.0], [1.0, 1.0]]);

    // lvl_1: node_ids are the *global representative ids* assigned to that root representative.
    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[0.0f32, 0.0]], // representative 0 (item 0)
        "node_ids",
        &array![0u32],
    );
    write_node(
        &store,
        "/lvl_1/node_1",
        &array![[1.0f32, 1.0], [10.0, 10.0], [11.0, 11.0]], // representatives 1, 2, 3 (items 2, 4, 6)
        "node_ids",
        &array![1u32, 2, 3],
    );

    // lvl_2: leaf level, so children are "item_ids" (dataset ids) instead.
    // Group numbers 0/1/2/3 line up with representative global ids 0/1/2/3 above.
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

    Index::load_from_store(as_readable_writable_listable(&store), None)
}

/// Writes a levels=1 (IVF-style) index, where root holds all 4
/// representatives and lvl_1 their leaf clusters, with the same 8 items as
/// `build_test_index`. Returns the store, so a test can load a second `Index`
/// from it to simulate a restart.
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

/// Loads `write_ivf_style_fixture`'s index.
pub(super) fn build_ivf_style_index() -> Index {
    Index::load_from_store(
        as_readable_writable_listable(&write_ivf_style_fixture()),
        None,
    )
}

/// Builds a 3-level tree, the only fixture where lvl_1's children are
/// internal nodes rather than leaves:
///   root:   2 entries -> lvl_1/node_0, lvl_1/node_1
///   lvl_1:  node_0 -> lvl_2/node_0, node_1    node_1 -> lvl_2/node_2, node_3
///   lvl_2:  node_i -> lvl_3/node_i
///   lvl_3:  node_i holds item i; items 0..3 at (0,0), (1,1), (10,10), (11,11)
///
/// Nearest-to-farthest from the origin is therefore item id order.
pub(super) fn build_three_level_test_index() -> Index {
    let store = new_memory_store();
    write_index_info(&store, 3, "L2", false);
    write_item_counts(&store, 4);
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

    Index::load_from_store(as_readable_writable_listable(&store), None)
}
