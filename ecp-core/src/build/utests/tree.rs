use super::*;
use crate::dtype::EmbeddingDtype;
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
use ndarray::array;
use std::sync::Barrier;
use std::thread;

#[test]
fn node_cache_evicts_once_over_capacity() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);
    append_node_batch(
        &store,
        "/lvl_1/node_0",
        "node_ids",
        &array![[1.0f32, 2.0]],
        &array![10u32],
        &[100, 2],
        EmbeddingDtype::F32,
    )
    .unwrap();
    append_node_batch(
        &store,
        "/lvl_1/node_1",
        "node_ids",
        &array![[3.0f32, 4.0]],
        &array![20u32],
        &[100, 2],
        EmbeddingDtype::F32,
    )
    .unwrap();

    // 1 embedding row (2 f32s) + 1 child id (1 u32) = 12 bytes, room for
    // exactly one entry, so caching both forces an eviction.
    let cache = NodeCache::new(12);
    cache.get_or_read(&store, "/lvl_1/node_0").unwrap();
    cache.get_or_read(&store, "/lvl_1/node_1").unwrap();
    cache.cache.run_pending_tasks();

    assert!(
        cache.cache.entry_count() < 2,
        "caching both entries under a one-entry capacity must have evicted one of them"
    );
    assert!(
        cache.cache.weighted_size() <= 12,
        "weighted size ({}) exceeded the 12-byte capacity",
        cache.cache.weighted_size()
    );
}

/// Threads released together by a barrier all ask for the same node, and
/// must all get the one cached entry.
#[test]
fn node_cache_get_or_read_is_safe_under_concurrent_access_to_the_same_path() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);
    append_node_batch(
        &store,
        "/lvl_1/node_0",
        "node_ids",
        &array![[1.0f32, 2.0]],
        &array![10u32],
        &[100, 2],
        EmbeddingDtype::F32,
    )
    .unwrap();

    let cache = NodeCache::new(1_000_000);
    const THREADS: usize = 8;
    let barrier = Barrier::new(THREADS);

    let results = thread::scope(|scope| {
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    cache.get_or_read(&store, "/lvl_1/node_0").unwrap().unwrap()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread panicked"))
            .collect::<Vec<_>>()
    });

    let first = &results[0];
    assert!(
        results.iter().all(|entry| Arc::ptr_eq(first, entry)),
        "every concurrent caller must share the same cached entry, not race into separate reads"
    );
}

/// Builds a tree with `build_tree` from in-memory sources, IP and not normalized.
fn build_ip_tree(
    store: &ReadableWritableListableStorage,
    representatives: Array2<f32>,
    dataset: Array2<f32>,
    total_levels: u32,
    node_size: usize,
) -> Result<()> {
    let root_embeddings = representatives.slice(s![..node_size, ..]).to_owned();
    build_tree(&BuildTreeArgs {
        store,
        root_embeddings: &root_embeddings,
        representatives: &EmbeddingsSource::Memory(representatives),
        dataset: &EmbeddingsSource::Memory(dataset),
        total_levels,
        metric: Metric::IP,
        is_normalized: false,
        fallback_batch_vecs: 100,
        chunk_shape: &[100, 2],
        embedding_dtype: EmbeddingDtype::F32,
        memory_limit_bytes: 1_000_000,
    })
}

/// Returns every item id held by the leaves `/lvl_{level}/node_0..node_count`, sorted.
fn leaf_item_ids(store: &ReadableWritableListableStorage, level: u32, node_count: u32) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..node_count)
        .flat_map(|node_id| {
            let node = Node::new(
                store.clone().readable_listable(),
                format!("/lvl_{level}/node_{node_id}"),
                "item_ids".to_string(),
            );
            node.children()
                .unwrap()
                .map(|children| children.to_vec())
                .unwrap_or_default()
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// Under IP, root entry 1 scores higher against root entry 0 than against
/// itself, so `lvl_1/node_1` is never written. Item 0 still scores highest
/// against root entry 1 and must fall back to root entry 0.
#[test]
fn build_tree_routes_past_a_never_written_internal_node() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);
    let representatives = array![[10.0f32, 0.0], [1.0, 0.1], [5.0, 1.0], [2.0, 1.0]];
    let dataset = array![[-1.0f32, 1.0], [3.0, 0.0]];

    build_ip_tree(&store, representatives, dataset, 2, 2).unwrap();

    assert_eq!(leaf_item_ids(&store, 2, 4), vec![0, 1]);
}

/// Representatives 2 and 3 are children of `lvl_1/node_0`, but no level-2
/// vector scores highest against either, so `lvl_2/node_2` and `lvl_2/node_3`
/// are never written. Item 0 scores highest against representative 2, then 3,
/// so it must fall back twice before reaching `lvl_2/node_0`.
#[test]
fn build_tree_falls_back_past_never_written_nodes_below_level_one() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);
    let representatives = array![
        [10.0f32, 0.0],
        [0.0, 10.0],
        [1.0, 0.1],
        [5.0, 1.0],
        [0.0, 5.0],
        [0.0, 6.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ];
    let dataset = array![[-1.0f32, -2.0], [3.0, 0.0], [0.0, 5.0]];

    build_ip_tree(&store, representatives, dataset, 3, 2).unwrap();

    assert_eq!(leaf_item_ids(&store, 3, 8), vec![0, 1, 2]);
}
