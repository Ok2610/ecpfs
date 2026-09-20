use super::*;
use crate::test_fixtures::{as_readable_listable, new_memory_store, write_node};
use ndarray::{Array1, Array2, array};

#[test]
fn loads_and_caches_embeddings_and_children() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
    let children: Array1<u32> = array![10, 20];
    write_node(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );

    assert!(!node.is_loaded());

    assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
    assert!(node.is_loaded());
    assert_eq!(node.children().as_ref().unwrap(), &children);
}

#[test]
fn resident_bytes_reflects_whats_actually_loaded() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
    let children: Array1<u32> = array![10, 20];
    write_node(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );
    assert_eq!(node.resident_bytes(), 0, "nothing loaded yet");

    node.embeddings();
    assert_eq!(node.resident_bytes(), 2 * 2 * 4, "2x2 f32 embeddings only");

    node.children();
    assert_eq!(
        node.resident_bytes(),
        2 * 2 * 4 + 2 * 4,
        "embeddings + 2 u32 children"
    );
}

#[test]
fn missing_node_yields_none_without_panicking() {
    let store = new_memory_store();
    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_absent".to_string(),
        "item_ids".to_string(),
    );

    assert!(node.embeddings().is_none());
    assert!(node.children().is_none());
    assert!(!node.is_loaded());
}

#[test]
fn a_confirmed_miss_is_cached_and_not_re_queried_after_data_appears() {
    let store = new_memory_store();
    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );

    assert!(node.embeddings().is_none());
    assert!(node.children().is_none());

    // The node exists now, but the cached `None` must stay.
    write_node(
        &store,
        "/lvl_1/node_0",
        &array![[1.0f32, 2.0]],
        "node_ids",
        &array![10u32],
    );

    assert!(
        node.embeddings().is_none(),
        "a confirmed miss must stay cached, not re-queried"
    );
    assert!(
        node.children().is_none(),
        "a confirmed miss must stay cached, not re-queried"
    );
}
