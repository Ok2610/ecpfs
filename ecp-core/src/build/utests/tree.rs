use super::*;
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
use ndarray::array;

#[test]
fn write_index_info_round_trips_through_index_load() {
    let store = new_memory_store();
    write_index_info(&as_readable_writable_listable(&store), 2, Metric::IP, true);

    let levels = Array::open(store.clone(), "/info/levels").expect("failed to open info/levels");
    assert_eq!(
        levels.retrieve_array_subset::<Vec<u32>>(&levels.subset_all()).expect("failed to read levels"),
        vec![2]
    );

    let metric = Array::open(store.clone(), "/info/metric").expect("failed to open info/metric");
    assert_eq!(
        metric.retrieve_array_subset::<Vec<String>>(&metric.subset_all()).expect("failed to read metric"),
        vec!["IP".to_string()]
    );

    let is_normalized =
        Array::open(store.clone(), "/info/is_normalized").expect("failed to open info/is_normalized");
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
    write_index_root(&as_readable_writable_listable(&store), &array![[1.0f32, 2.0], [3.0, 4.0]], &[100, 2]);

    let root = Array::open(store.clone(), "/index_root/embeddings").expect("failed to open index_root/embeddings");
    assert_eq!(
        root.retrieve_array_subset::<Array2<f32>>(&root.subset_all()).expect("failed to read root"),
        array![[1.0f32, 2.0], [3.0, 4.0]]
    );
}

#[test]
fn append_node_batch_creates_embeddings_children_and_a_border_placeholder() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    append_node_batch(&store, "/lvl_1/node_0", "item_ids", &array![[1.0f32, 2.0]], &array![10u32], &[100, 2]);

    let embeddings = Array::open(store.clone(), "/lvl_1/node_0/embeddings").expect("failed to open embeddings");
    assert_eq!(
        embeddings.retrieve_array_subset::<Array2<f32>>(&embeddings.subset_all()).expect("failed to read embeddings"),
        array![[1.0f32, 2.0]]
    );

    let ids = Array::open(store.clone(), "/lvl_1/node_0/item_ids").expect("failed to open item_ids");
    assert_eq!(
        ids.retrieve_array_subset::<Array1<u32>>(&ids.subset_all()).expect("failed to read item_ids"),
        array![10u32]
    );

    let border = Array::open(store.clone(), "/lvl_1/node_0/border").expect("border placeholder must exist");
    assert_eq!(border.shape(), &[2]);
}

#[test]
fn append_node_batch_grows_an_existing_node_across_multiple_calls() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    append_node_batch(&store, "/lvl_1/node_0", "item_ids", &array![[1.0f32, 2.0]], &array![10u32], &[100, 2]);
    append_node_batch(&store, "/lvl_1/node_0", "item_ids", &array![[3.0f32, 4.0]], &array![20u32], &[100, 2]);

    let embeddings = Array::open(store.clone(), "/lvl_1/node_0/embeddings").expect("failed to open embeddings");
    assert_eq!(
        embeddings.retrieve_array_subset::<Array2<f32>>(&embeddings.subset_all()).expect("failed to read embeddings"),
        array![[1.0f32, 2.0], [3.0, 4.0]]
    );

    let ids = Array::open(store.clone(), "/lvl_1/node_0/item_ids").expect("failed to open item_ids");
    assert_eq!(
        ids.retrieve_array_subset::<Array1<u32>>(&ids.subset_all()).expect("failed to read item_ids"),
        array![10u32, 20]
    );
}
