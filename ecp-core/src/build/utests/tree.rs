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
fn write_info_u32_stores_a_named_scalar() {
    let store = new_memory_store();
    write_info_u32(&as_readable_writable_listable(&store), "total_items", 42);

    let total_items =
        Array::open(store.clone(), "/info/total_items").expect("failed to open info/total_items");
    assert_eq!(
        total_items
            .retrieve_array_subset::<Vec<u32>>(&total_items.subset_all())
            .expect("failed to read total_items"),
        vec![42]
    );
}

#[test]
fn write_info_u32_overwrites_an_existing_scalar() {
    let store = new_memory_store();
    let writable = as_readable_writable_listable(&store);
    write_info_u32(&writable, "next_item_id", 42);
    write_info_u32(&writable, "next_item_id", 99);

    let field =
        Array::open(store.clone(), "/info/next_item_id").expect("failed to open info/next_item_id");
    assert_eq!(
        field
            .retrieve_array_subset::<Vec<u32>>(&field.subset_all())
            .expect("failed to read next_item_id"),
        vec![99],
        "insert rewrites these fields in place on every call"
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
fn node_cache_holds_an_entry_within_its_capacity() {
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

    let cache = NodeCache::new(1_000_000);
    let first = cache.get_or_read(&store, "/lvl_1/node_0");
    let second = cache.get_or_read(&store, "/lvl_1/node_0");

    assert!(
        Arc::ptr_eq(&first, &second),
        "a second lookup within capacity must return the same cached entry"
    );
}

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

    // 1 embedding row (2 f32s) + 1 child id (1 u32) = 12 bytes: room for
    // exactly one entry, so caching both forces an eviction.
    let cache = NodeCache::new(12);
    cache.get_or_read(&store, "/lvl_1/node_0");
    cache.get_or_read(&store, "/lvl_1/node_1");
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

/// A barrier forces every thread to call `get_or_read` at once, so moka's
/// own single-flight get-or-insert (`get_with`) actually gets exercised
/// under real concurrent contention on the same key.
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

    let cache = NodeCache::new(1_000_000);
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
}
