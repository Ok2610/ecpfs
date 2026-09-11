use super::*;
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
use crate::utils::EmbeddingDtype;
use half::f16;
use ndarray::array;

#[test]
fn zarrs_append_creates_on_first_call_and_grows_on_later_calls() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[1.0f32, 2.0], [3.0, 4.0]],
        &array![10u32, 20],
        &[100, 2],
        EmbeddingDtype::F32,
    );
    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[5.0f32, 6.0]],
        &array![30u32],
        &[100, 2],
        EmbeddingDtype::F32,
    );

    let embeddings =
        Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
    let ids = Array::open(store.clone(), "/node/item_ids").expect("failed to open ids");

    assert_eq!(
        embeddings
            .retrieve_array_subset::<Array2<f32>>(&embeddings.subset_all())
            .expect("failed to read embeddings"),
        array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]
    );
    assert_eq!(
        ids.retrieve_array_subset::<Array1<u32>>(&ids.subset_all())
            .expect("failed to read ids"),
        array![10u32, 20, 30]
    );
}

#[test]
fn zarrs_append_ignores_chunk_shape_on_a_later_call() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[1.0f32, 2.0], [3.0, 4.0]],
        &array![10u32, 20],
        &[100, 2],
        EmbeddingDtype::F32,
    );
    // A different chunk_shape here must have no effect: the array already exists.
    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[5.0f32, 6.0]],
        &array![30u32],
        &[7, 2],
        EmbeddingDtype::F32,
    );

    let embeddings =
        Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
    assert_eq!(
        embeddings
            .chunk_shape_usize(&[0, 0])
            .expect("failed to read chunk shape"),
        vec![100, 2]
    );
    assert_eq!(
        embeddings
            .retrieve_array_subset::<Array2<f32>>(&embeddings.subset_all())
            .expect("failed to read embeddings"),
        array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]
    );
}

#[test]
fn zarrs_append_writes_f16_when_requested() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[1.5f32, 2.5]],
        &array![10u32],
        &[100, 2],
        EmbeddingDtype::F16,
    );
    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[3.5f32, 4.5]],
        &array![20u32],
        &[100, 2],
        EmbeddingDtype::F16,
    );

    let embeddings =
        Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
    assert_eq!(*embeddings.data_type(), zarrs::array::data_type::float16());
    assert_eq!(
        embeddings
            .retrieve_array_subset::<Array2<f16>>(&embeddings.subset_all())
            .expect("failed to read embeddings"),
        array![
            [f16::from_f32(1.5), f16::from_f32(2.5)],
            [f16::from_f32(3.5), f16::from_f32(4.5)]
        ]
    );
}
