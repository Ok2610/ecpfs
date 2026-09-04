use super::*;
use crate::test_fixtures::{as_readable_listable, new_memory_store, write_node, write_node_f16, write_node_unsupported_dtype};
use ndarray::{array, Array1, Array2};

#[test]
fn loads_and_caches_embeddings_and_children() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
    let children: Array1<u32> = array![10, 20];
    write_node(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let mut node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );

    assert!(!node.is_loaded());

    assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
    assert!(node.is_loaded());
    assert_eq!(node.children().as_ref().unwrap(), &children);

    node.clear_cache();
    assert!(!node.is_loaded());
    // Lazily reloads from the store after a cache clear.
    assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
}

/// Values chosen to be exactly representable in f16 (10 mantissa bits), so the
/// upcast to f32 via `mapv(|x: f16| x.to_f32())` is exact, not approximate.
#[test]
fn loads_f16_embeddings_upcast_to_f32() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[1.0, 2.0], [3.5, -1.25]];
    let children: Array1<u32> = array![10, 20];
    write_node_f16(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let mut node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_0".to_string(),
        "node_ids".to_string(),
    );

    assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
}

/// f16 and f32 are the only supported embedding dtypes; anything else (e.g. an
/// embeddings array left as float64 by an upstream caller that never cast it)
/// must fail loudly at load time rather than silently truncating precision.
#[test]
#[should_panic(expected = "unknown datatype")]
fn unsupported_dtype_panics_instead_of_silently_truncating() {
    let store = new_memory_store();
    let embeddings: Array2<f32> = array![[1.0, 2.0], [3.5, -1.25]];
    let children: Array1<u32> = array![10, 20];
    write_node_unsupported_dtype(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

    let mut node = Node::new(
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

    let mut node = Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "node_ids".to_string());
    assert_eq!(node.resident_bytes(), 0, "nothing loaded yet");

    node.embeddings();
    assert_eq!(node.resident_bytes(), 2 * 2 * 4, "2x2 f32 embeddings only");

    node.children();
    assert_eq!(node.resident_bytes(), 2 * 2 * 4 + 2 * 4, "embeddings + 2 u32 children");

    node.clear_cache();
    assert_eq!(node.resident_bytes(), 0);
}

#[test]
fn missing_node_yields_none_without_panicking() {
    let store = new_memory_store();
    let mut node = Node::new(
        as_readable_listable(&store),
        "/lvl_1/node_absent".to_string(),
        "item_ids".to_string(),
    );

    assert!(node.embeddings().is_none());
    assert!(node.children().is_none());
    assert!(!node.is_loaded());
}
