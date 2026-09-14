use super::*;
use crate::test_fixtures::{
    as_readable_listable, new_memory_store, write_node, write_node_f16, write_node_uint8,
    write_node_unsupported_dtype,
};
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

/// Values chosen to be exactly representable in f16 (10 mantissa bits), so the
/// upcast to f32 via `mapv(|x: f16| x.to_f32())` is exact, not approximate.
#[test]
fn loads_f16_embeddings_upcast_to_f32() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[1.0, 2.0], [3.5, -1.25]];
    let children: Array1<u32> = array![10, 20];
    write_node_f16(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );

    assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
}

/// Unlike the f16 case, the widening here is exact for every possible stored
/// value: f32 represents all 256 uint8 values without rounding.
#[test]
fn loads_uint8_embeddings_widened_to_f32() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[0.0, 255.0], [1.0, 128.0]];
    let children: Array1<u32> = array![10, 20];
    write_node_uint8(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );

    assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
}

/// Everything is cached as f32 whatever the stored width, so a uint8 node
/// accounts for 4 bytes per value, not 1. The memory budget is sized off
/// resident bytes, so reporting the on-disk width here would overcommit it.
#[test]
fn resident_bytes_counts_the_widened_f32_size_not_the_stored_width() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[0.0, 255.0], [1.0, 128.0]];
    let children: Array1<u32> = array![10, 20];
    write_node_uint8(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );
    node.embeddings();
    node.children();

    assert_eq!(
        node.resident_bytes(),
        4 * size_of::<f32>() + 2 * size_of::<u32>()
    );
}

/// Anything outside the supported dtypes (e.g. an embeddings array left as
/// float64 by an upstream caller that never cast it) must fail loudly at
/// load time rather than silently truncating precision.
#[test]
#[should_panic(expected = "unsupported embeddings dtype")]
fn unsupported_dtype_panics_instead_of_silently_truncating() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[1.0, 2.0], [3.5, -1.25]];
    let children: Array1<u32> = array![10, 20];
    write_node_unsupported_dtype(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );

    node.embeddings();
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

    // The node now actually exists; a naive `is_none()`-only check would
    // re-query and find it, but a `OnceLock` already resolved to `None` must not.
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
