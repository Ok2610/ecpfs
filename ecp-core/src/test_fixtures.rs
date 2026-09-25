#![cfg(test)]

use std::sync::Arc;

use ndarray::{Array1, Array2};
use zarrs::array::data_type::{bool, float32, string, uint32};
use zarrs::array::{ArrayBuilder, FillValueMetadata};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    ReadableListableStorage, ReadableWritableListableStorage, StorePrefix, WritableStorageTraits,
};

/// Writes `children` as the node's `child_key` array (`node_ids` or `item_ids`).
fn write_children(
    store: &Arc<MemoryStore>,
    group_path: &str,
    child_key: &str,
    children: &Array1<u32>,
) {
    let child_path = format!("{group_path}/{child_key}");
    let child_shape = vec![children.len() as u64];
    let child_array = ArrayBuilder::new(child_shape.clone(), child_shape, uint32(), 0u32)
        .build(store.clone(), &child_path)
        .expect("failed to build children array");
    child_array
        .store_metadata()
        .expect("failed to store children metadata");
    child_array
        .store_chunk(&[0], children)
        .expect("failed to store children chunk");
}

/// Writes a node's `embeddings` and `children` arrays at `group_path` in an
/// in-memory store, the same layout `append_node_batch` writes. `child_key`
/// names the children array (`node_ids` or `item_ids`).
pub fn write_node(
    store: &Arc<MemoryStore>,
    group_path: &str,
    embeddings: &Array2<f32>,
    child_key: &str,
    children: &Array1<u32>,
) {
    let emb_path = format!("{group_path}/embeddings");
    let emb_shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let emb_array = ArrayBuilder::new(emb_shape.clone(), emb_shape, float32(), 0.0f32)
        .build(store.clone(), &emb_path)
        .expect("failed to build embeddings array");
    emb_array
        .store_metadata()
        .expect("failed to store embeddings metadata");
    emb_array
        .store_chunk(&[0, 0], embeddings)
        .expect("failed to store embeddings chunk");

    write_children(store, group_path, child_key, children);
}

/// Writes `info/levels`, `info/metric` and `info/is_normalized` the same way
/// the build's `write_index_info` does.
pub fn write_index_info(store: &Arc<MemoryStore>, levels: u32, metric: &str, is_normalized: bool) {
    let scalar_shape: Vec<u64> = vec![];

    let levels_array =
        ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), uint32(), 0u32)
            .build(store.clone(), "/info/levels")
            .expect("failed to build info/levels array");
    levels_array
        .store_metadata()
        .expect("failed to store info/levels metadata");
    levels_array
        .store_chunk(&[], vec![levels])
        .expect("failed to store info/levels chunk");

    let metric_array = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), string(), "")
        .build(store.clone(), "/info/metric")
        .expect("failed to build info/metric array");
    metric_array
        .store_metadata()
        .expect("failed to store info/metric metadata");
    metric_array
        .store_chunk(&[], vec![metric.to_string()])
        .expect("failed to store info/metric chunk");

    let is_normalized_array = ArrayBuilder::new(
        scalar_shape.clone(),
        scalar_shape,
        bool(),
        FillValueMetadata::Bool(false),
    )
    .build(store.clone(), "/info/is_normalized")
    .expect("failed to build info/is_normalized array");
    is_normalized_array
        .store_metadata()
        .expect("failed to store info/is_normalized metadata");
    is_normalized_array
        .store_chunk(&[], vec![is_normalized])
        .expect("failed to store info/is_normalized chunk");
}

/// Writes `info/{name}` as a `u32` scalar, the same way the build's
/// `write_info_u32` does.
pub fn write_info_u32(store: &Arc<MemoryStore>, name: &str, value: u32) {
    let scalar_shape: Vec<u64> = vec![];
    let path = format!("/info/{name}");
    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape, uint32(), 0u32)
        .build(store.clone(), &path)
        .unwrap_or_else(|e| panic!("failed to build {path} array: {e}"));
    field
        .store_metadata()
        .unwrap_or_else(|e| panic!("failed to store {path} metadata: {e}"));
    field
        .store_chunk(&[], vec![value])
        .unwrap_or_else(|e| panic!("failed to store {path} chunk: {e}"));
}

/// Writes `info/total_items` and `info/next_item_id`, both set to
/// `total_items` as a fresh build leaves them. A test simulating a crashed
/// insert overwrites `next_item_id` afterwards.
pub fn write_item_counts(store: &Arc<MemoryStore>, total_items: u32) {
    write_info_u32(store, "total_items", total_items);
    write_info_u32(store, "next_item_id", total_items);
}

/// Removes `info/{name}`, leaving the index as one built without that field.
pub fn erase_info_field(store: &Arc<MemoryStore>, name: &str) {
    let prefix = StorePrefix::new(format!("info/{name}/")).expect("info field prefix is valid");
    store
        .erase_prefix(&prefix)
        .unwrap_or_else(|e| panic!("failed to erase info/{name}: {e}"));
}

/// Writes `ids` as `/rep_item_ids`, the representative ids a build saves.
pub fn write_rep_item_ids(store: &Arc<MemoryStore>, ids: &Array1<u32>) {
    let shape = vec![ids.len() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, uint32(), 0u32)
        .build(store.clone(), "/rep_item_ids")
        .expect("failed to build rep_item_ids array");
    array
        .store_metadata()
        .expect("failed to store rep_item_ids metadata");
    array
        .store_chunk(&[0], ids)
        .expect("failed to store rep_item_ids chunk");
}

/// Writes `embeddings` as `index_root/embeddings`, the root node's representatives.
pub fn write_index_root(store: &Arc<MemoryStore>, embeddings: &Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let root_array = ArrayBuilder::new(shape.clone(), shape, float32(), 0.0f32)
        .build(store.clone(), "/index_root/embeddings")
        .expect("failed to build index_root/embeddings array");
    root_array
        .store_metadata()
        .expect("failed to store index_root/embeddings metadata");
    root_array
        .store_chunk(&[0, 0], embeddings)
        .expect("failed to store index_root/embeddings chunk");
}

/// Creates an empty in-memory zarr store.
pub fn new_memory_store() -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new())
}

/// Returns `store` as read-only zarr storage.
pub fn as_readable_listable(store: &Arc<MemoryStore>) -> ReadableListableStorage {
    store.clone()
}

/// Returns `store` as read-write zarr storage.
pub fn as_readable_writable_listable(store: &Arc<MemoryStore>) -> ReadableWritableListableStorage {
    store.clone()
}
