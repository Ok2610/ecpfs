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
                    cache.get_or_read(&store, "/lvl_1/node_0").unwrap()
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
