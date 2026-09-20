use super::*;
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
use ndarray::array;
use zarrs::array::data_type::float32;
use zarrs::array::{Array, ArrayBuilder};
use zarrs::storage::store::MemoryStore;

#[test]
fn resolve_dtype_uses_embedding_dtype_when_set_else_native() {
    assert_eq!(
        resolve_dtype(None, EmbeddingDtype::F16),
        EmbeddingDtype::F16
    );
    assert_eq!(
        resolve_dtype(Some(EmbeddingDtype::UInt8), EmbeddingDtype::F32),
        EmbeddingDtype::UInt8,
        "a set embedding_dtype wins even when it can't hold every source value"
    );
}

/// Writes `embeddings` as a one-chunk f32 array at `path` and returns a
/// source over it.
fn write_source(
    store: &std::sync::Arc<MemoryStore>,
    path: &str,
    embeddings: &Array2<f32>,
) -> EmbeddingsSource {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape.clone(), float32(), 0.0f32)
        .build(store.clone(), path)
        .expect("failed to build source array");
    array
        .store_metadata()
        .expect("failed to store source metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[0..shape[0], 0..shape[1]]),
            embeddings,
        )
        .expect("failed to store source embeddings");

    EmbeddingsSource::from_zarr(store.clone(), path.to_string())
}

/// Makes an L2 `Builder` on `store` with the default chunk size and no
/// dtype override.
fn new_builder(
    store: &std::sync::Arc<MemoryStore>,
    levels: u32,
    memory_limit_bytes: usize,
) -> Builder {
    Builder::new(
        as_readable_writable_listable(store),
        levels,
        Metric::L2,
        false,
        memory_limit_bytes,
        None,
        DEFAULT_MAX_CHUNK_BYTES,
    )
}

#[test]
fn select_representatives_uses_the_builders_max_chunk_bytes() {
    let store = new_memory_store();
    let source = write_source(
        &store,
        "/dataset",
        &array![[0.0f32, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    );
    // dim=2 -> 8 bytes/vec; a 16-byte budget fits exactly 2 vecs per chunk.
    let mut builder = Builder::new(
        as_readable_writable_listable(&store),
        2,
        Metric::L2,
        false,
        1_000_000,
        None,
        16,
    );

    builder.select_representatives(&source, 3, RepresentativeStrategy::Offset, 10);

    let rep_embeddings = Array::open(as_readable_writable_listable(&store), "/rep_embeddings")
        .expect("failed to open rep_embeddings");
    assert_eq!(
        rep_embeddings
            .chunk_shape_usize(&[0, 0])
            .expect("failed to read chunk shape"),
        vec![2, 2]
    );
}

/// select_representatives saves the representatives to disk and keeps no
/// copy in memory, even when memory_limit_bytes has room for one.
#[test]
fn select_representatives_sets_node_size_and_persists_only() {
    let store = new_memory_store();
    let source = write_source(
        &store,
        "/dataset",
        &array![[0.0f32, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    );
    let mut builder = new_builder(&store, 2, 1_000_000);

    builder.select_representatives(&source, 3, RepresentativeStrategy::Offset, 10);

    assert_eq!(
        builder.node_size, 2,
        "4 items / 3 per cluster -> 2 representatives, ceil(2^(1/2)) = 2"
    );
    assert!(matches!(
        builder.representatives,
        Some(Representatives::PersistedOnly)
    ));
}

#[test]
fn select_representatives_custom_uses_caller_supplied_representatives() {
    let store = new_memory_store();
    let mut builder = new_builder(&store, 1, 1_000_000);

    builder.select_representatives_custom(
        Array1::from_vec(vec![7u32, 9]),
        array![[0.0f32, 0.0], [5.0, 5.0]],
    );

    assert_eq!(builder.node_size, 2);
    match builder.representatives {
        Some(Representatives::InMemory { ref ids, .. }) => assert_eq!(ids.to_vec(), vec![7, 9]),
        _ => panic!("expected the custom representatives to be kept in memory"),
    }
}

#[test]
#[should_panic(expected = "ids and embeddings must have the same length")]
fn select_representatives_custom_panics_on_a_length_mismatch() {
    let store = new_memory_store();
    let mut builder = new_builder(&store, 1, 1_000_000);

    builder.select_representatives_custom(
        Array1::from_vec(vec![7u32, 9, 11]),
        array![[0.0f32, 0.0], [5.0, 5.0]],
    );
}

#[test]
#[should_panic(expected = "call select_representatives before build")]
fn build_without_representatives_panics() {
    let store = new_memory_store();
    let source = write_source(&store, "/dataset", &array![[0.0f32, 0.0]]);
    let mut builder = new_builder(&store, 1, 1_000_000);

    builder.build(&source, 10);
}

/// With levels=1 the root's children are the leaves, so only one level is
/// built. Two well-separated pairs of points; `Offset` picks vecs 0 and 2.
#[test]
fn build_writes_index_root_and_leaf_nodes() {
    let store = new_memory_store();
    let dataset = write_source(
        &store,
        "/dataset",
        &array![[0.0f32, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]],
    );
    let mut builder = new_builder(&store, 1, 1_000_000);

    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 10);
    builder.build(&dataset, 10);

    let root = Array::open(
        as_readable_writable_listable(&store),
        "/index_root/embeddings",
    )
    .unwrap();
    assert_eq!(
        root.retrieve_array_subset::<Array2<f32>>(&root.subset_all())
            .unwrap(),
        array![[0.0f32, 0.0], [10.0, 0.0]]
    );

    let node_0_ids = Array::open(
        as_readable_writable_listable(&store),
        "/lvl_1/node_0/item_ids",
    )
    .unwrap();
    assert_eq!(
        node_0_ids
            .retrieve_array_subset::<Array1<u32>>(&node_0_ids.subset_all())
            .unwrap(),
        array![0u32, 1]
    );

    let node_1_ids = Array::open(
        as_readable_writable_listable(&store),
        "/lvl_1/node_1/item_ids",
    )
    .unwrap();
    assert_eq!(
        node_1_ids
            .retrieve_array_subset::<Array1<u32>>(&node_1_ids.subset_all())
            .unwrap(),
        array![2u32, 3]
    );
}

#[test]
fn build_writes_total_items_and_next_item_id_from_the_datasets_row_count() {
    let store = new_memory_store();
    let dataset = write_source(
        &store,
        "/dataset",
        &array![[0.0f32, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]],
    );
    let mut builder = new_builder(&store, 1, 1_000_000);

    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 10);
    builder.build(&dataset, 10);

    for field in ["total_items", "next_item_id"] {
        let array = Array::open(
            as_readable_writable_listable(&store),
            &format!("/info/{field}"),
        )
        .unwrap();
        assert_eq!(
            array
                .retrieve_array_subset::<Vec<u32>>(&array.subset_all())
                .unwrap(),
            vec![4],
            "{field}"
        );
    }
}

/// Exercises `build`'s `Representatives::InMemory` branch.
#[test]
fn build_with_custom_representatives_writes_index_root_and_leaf_nodes() {
    let store = new_memory_store();
    let dataset = write_source(
        &store,
        "/dataset",
        &array![[0.0f32, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]],
    );
    let mut builder = new_builder(&store, 1, 1_000_000);

    builder.select_representatives_custom(
        Array1::from_vec(vec![7u32, 9u32]),
        array![[0.0f32, 0.0], [10.0, 0.0]],
    );
    builder.build(&dataset, 10);

    let root = Array::open(
        as_readable_writable_listable(&store),
        "/index_root/embeddings",
    )
    .unwrap();
    assert_eq!(
        root.retrieve_array_subset::<Array2<f32>>(&root.subset_all())
            .unwrap(),
        array![[0.0f32, 0.0], [10.0, 0.0]]
    );

    let node_0_ids = Array::open(
        as_readable_writable_listable(&store),
        "/lvl_1/node_0/item_ids",
    )
    .unwrap();
    assert_eq!(
        node_0_ids
            .retrieve_array_subset::<Array1<u32>>(&node_0_ids.subset_all())
            .unwrap(),
        array![0u32, 1]
    );

    let node_1_ids = Array::open(
        as_readable_writable_listable(&store),
        "/lvl_1/node_1/item_ids",
    )
    .unwrap();
    assert_eq!(
        node_1_ids
            .retrieve_array_subset::<Array1<u32>>(&node_1_ids.subset_all())
            .unwrap(),
        array![2u32, 3]
    );
}
