use super::*;
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
use crate::utils::EmbeddingDtype;
use ndarray::array;
use std::sync::Barrier;
use std::thread;

#[test]
fn write_index_info_round_trips_through_index_load() {
    let store = new_memory_store();
    write_index_info(&as_readable_writable_listable(&store), 2, Metric::IP, true);

    let levels = Array::open(store.clone(), "/info/levels").expect("failed to open info/levels");
    assert_eq!(
        levels
            .retrieve_array_subset::<Vec<u32>>(&levels.subset_all())
            .expect("failed to read levels"),
        vec![2]
    );

    let metric = Array::open(store.clone(), "/info/metric").expect("failed to open info/metric");
    assert_eq!(
        metric
            .retrieve_array_subset::<Vec<String>>(&metric.subset_all())
            .expect("failed to read metric"),
        vec!["IP".to_string()]
    );

    let is_normalized = Array::open(store.clone(), "/info/is_normalized")
        .expect("failed to open info/is_normalized");
    assert_eq!(
        is_normalized
            .retrieve_array_subset::<Vec<bool>>(&is_normalized.subset_all())
            .expect("failed to read is_normalized"),
        vec![true]
    );
}

#[test]
fn write_index_root_stores_the_leader_embeddings() {
    let store = new_memory_store();
    write_index_root(
        &as_readable_writable_listable(&store),
        &array![[1.0f32, 2.0], [3.0, 4.0]],
        &[100, 2],
        EmbeddingDtype::F32,
    );

    let root = Array::open(store.clone(), "/index_root/embeddings")
        .expect("failed to open index_root/embeddings");
    assert_eq!(
        root.retrieve_array_subset::<Array2<f32>>(&root.subset_all())
            .expect("failed to read root"),
        array![[1.0f32, 2.0], [3.0, 4.0]]
    );
}

#[test]
fn append_node_batch_creates_embeddings_children_and_a_border_placeholder() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    append_node_batch(
        &store,
        "/lvl_1/node_0",
        "item_ids",
        &array![[1.0f32, 2.0]],
        &array![10u32],
        &[100, 2],
        EmbeddingDtype::F32,
    );

    let embeddings =
        Array::open(store.clone(), "/lvl_1/node_0/embeddings").expect("failed to open embeddings");
    assert_eq!(
        embeddings
            .retrieve_array_subset::<Array2<f32>>(&embeddings.subset_all())
            .expect("failed to read embeddings"),
        array![[1.0f32, 2.0]]
    );

    let ids =
        Array::open(store.clone(), "/lvl_1/node_0/item_ids").expect("failed to open item_ids");
    assert_eq!(
        ids.retrieve_array_subset::<Array1<u32>>(&ids.subset_all())
            .expect("failed to read item_ids"),
        array![10u32]
    );

    let border =
        Array::open(store.clone(), "/lvl_1/node_0/border").expect("border placeholder must exist");
    assert_eq!(border.shape(), &[2]);
}

#[test]
fn append_node_batch_grows_an_existing_node_across_multiple_calls() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    append_node_batch(
        &store,
        "/lvl_1/node_0",
        "item_ids",
        &array![[1.0f32, 2.0]],
        &array![10u32],
        &[100, 2],
        EmbeddingDtype::F32,
    );
    append_node_batch(
        &store,
        "/lvl_1/node_0",
        "item_ids",
        &array![[3.0f32, 4.0]],
        &array![20u32],
        &[100, 2],
        EmbeddingDtype::F32,
    );

    let embeddings =
        Array::open(store.clone(), "/lvl_1/node_0/embeddings").expect("failed to open embeddings");
    assert_eq!(
        embeddings
            .retrieve_array_subset::<Array2<f32>>(&embeddings.subset_all())
            .expect("failed to read embeddings"),
        array![[1.0f32, 2.0], [3.0, 4.0]]
    );

    let ids =
        Array::open(store.clone(), "/lvl_1/node_0/item_ids").expect("failed to open item_ids");
    assert_eq!(
        ids.retrieve_array_subset::<Array1<u32>>(&ids.subset_all())
            .expect("failed to read item_ids"),
        array![10u32, 20]
    );
}

#[test]
fn node_cache_is_unbounded_until_set_limit_is_called() {
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
    );

    let cache = NodeCache::new();
    let first = cache.get_or_read(&store, "/lvl_1/node_0");
    let second = cache.get_or_read(&store, "/lvl_1/node_0");

    assert!(
        Arc::ptr_eq(&first, &second),
        "an entry must survive without set_limit ever being called"
    );
}

#[test]
fn node_cache_set_limit_evicts_the_least_recently_used_entry() {
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
    );
    append_node_batch(
        &store,
        "/lvl_1/node_1",
        "node_ids",
        &array![[3.0f32, 4.0]],
        &array![20u32],
        &[100, 2],
        EmbeddingDtype::F32,
    );

    let cache = NodeCache::new();
    let node_0_first = cache.get_or_read(&store, "/lvl_1/node_0");
    let node_1_first = cache.get_or_read(&store, "/lvl_1/node_1");
    // Touch node_0 again so node_1 becomes the least-recently-used entry.
    cache.get_or_read(&store, "/lvl_1/node_0");

    // 1 embedding row (2 f32s) + 1 child id (1 u32) = 12 bytes: room for exactly one entry.
    cache.set_limit(12);

    let node_0_after = cache.get_or_read(&store, "/lvl_1/node_0");
    assert!(
        Arc::ptr_eq(&node_0_first, &node_0_after),
        "node_0 was touched more recently, so it must survive eviction"
    );

    let node_1_after = cache.get_or_read(&store, "/lvl_1/node_1");
    assert!(
        !Arc::ptr_eq(&node_1_first, &node_1_after),
        "node_1 was the least-recently-used entry, so it must have been evicted"
    );
}

/// `get_or_read` checks the cache, releases the lock, reads from disk, then
/// re-checks the cache before inserting, so a losing thread's read never
/// gets counted. A barrier forces every thread to call `get_or_read` at
/// once, so that race window actually gets exercised.
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
    );

    let cache = NodeCache::new();
    const THREADS: usize = 8;
    let barrier = Barrier::new(THREADS);

    let results = thread::scope(|scope| {
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    cache.get_or_read(&store, "/lvl_1/node_0")
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

    // If a losing thread's read had been double-counted into the cache's
    // byte total, a limit sized for exactly one entry would evict it immediately.
    cache.set_limit(12);
    let after = cache.get_or_read(&store, "/lvl_1/node_0");
    assert!(
        Arc::ptr_eq(first, &after),
        "byte accounting must not be inflated by concurrent misses racing on the same entry"
    );
}
