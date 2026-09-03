use super::*;
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
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
    );
    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[5.0f32, 6.0]],
        &array![30u32],
        &[100, 2],
    );

    let embeddings = Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
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
