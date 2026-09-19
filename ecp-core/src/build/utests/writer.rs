use super::*;
use crate::dtype::{EmbeddingDtype, read_subset_as_f32};
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
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

/// Unlike the f16 case above, integer-valued data survives a uint8 round
/// trip exactly, which is the whole point of offering the dtype.
#[test]
fn zarrs_append_round_trips_integer_values_through_uint8_exactly() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[0.0f32, 255.0], [1.0, 128.0]],
        &array![10u32, 20],
        &[100, 2],
        EmbeddingDtype::UInt8,
    );

    let embeddings =
        Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
    assert_eq!(*embeddings.data_type(), zarrs::array::data_type::uint8());
    assert_eq!(
        read_subset_as_f32(&embeddings, &embeddings.subset_all(), "embeddings"),
        array![[0.0f32, 255.0], [1.0, 128.0]],
        "every stored value must read back bit-for-bit as the f32 it went in as"
    );
}

#[test]
fn zarrs_append_round_trips_integer_values_through_int8_exactly() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[-128.0f32, 127.0], [-1.0, 0.0]],
        &array![10u32, 20],
        &[100, 2],
        EmbeddingDtype::Int8,
    );

    let embeddings =
        Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
    assert_eq!(*embeddings.data_type(), zarrs::array::data_type::int8());
    assert_eq!(
        read_subset_as_f32(&embeddings, &embeddings.subset_all(), "embeddings"),
        array![[-128.0f32, 127.0], [-1.0, 0.0]]
    );
}

/// Rust's float-to-int `as` saturates rather than wrapping, so an
/// out-of-range value clamps to the nearest end and NaN becomes 0. Pinned
/// as a contract: the write path relies on it instead of range-checking,
/// and `resolve_dtype` only warns about it.
#[test]
fn storing_out_of_range_values_as_uint8_saturates_rather_than_wrapping() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[300.0f32, -5.0], [f32::NAN, 2.7]],
        &array![10u32, 20],
        &[100, 2],
        EmbeddingDtype::UInt8,
    );

    let embeddings =
        Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
    assert_eq!(
        read_subset_as_f32(&embeddings, &embeddings.subset_all(), "embeddings"),
        array![[255.0f32, 0.0], [0.0, 2.0]],
        "300 clamps to 255, -5 clamps to 0, NaN becomes 0, and 2.7 truncates to 2"
    );
}

#[test]
fn storing_out_of_range_values_as_int8_saturates_rather_than_wrapping() {
    let store = new_memory_store();
    let store = as_readable_writable_listable(&store);

    zarrs_append(
        &store,
        "/node/embeddings",
        "/node/item_ids",
        &array![[200.0f32, -200.0]],
        &array![10u32],
        &[100, 2],
        EmbeddingDtype::Int8,
    );

    let embeddings =
        Array::open(store.clone(), "/node/embeddings").expect("failed to open embeddings");
    assert_eq!(
        read_subset_as_f32(&embeddings, &embeddings.subset_all(), "embeddings"),
        array![[127.0f32, -128.0]]
    );
}

#[test]
fn write_index_info_stores_levels_metric_and_is_normalized() {
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
