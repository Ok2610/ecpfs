#![cfg(test)]

use std::sync::Arc;

use half::f16;
use ndarray::{Array1, Array2};
use zarrs::array::data_type::{bool, float16, float32, float64, string, uint8, uint32};
use zarrs::array::{ArrayBuilder, FillValueMetadata};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

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

/// Writes a node's `embeddings` array and its `child_key` (node_ids/item_ids) array
/// into `group_path` on an in-memory zarr store, mirroring what the Python builder
/// writes to disk for a real index.
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

/// Like `write_node`, but stores `embeddings` as zarr's `float16` dtype, so it
/// exercises `Node::embeddings()`'s f16-upcast branch, which `write_node`
/// (always float32) never reaches.
pub fn write_node_f16(
    store: &Arc<MemoryStore>,
    group_path: &str,
    embeddings: &Array2<f32>,
    child_key: &str,
    children: &Array1<u32>,
) {
    let embeddings_f16 = embeddings.mapv(f16::from_f32);
    let emb_path = format!("{group_path}/embeddings");
    let emb_shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let emb_array = ArrayBuilder::new(emb_shape.clone(), emb_shape, float16(), f16::from_f32(0.0))
        .build(store.clone(), &emb_path)
        .expect("failed to build embeddings array");
    emb_array
        .store_metadata()
        .expect("failed to store embeddings metadata");
    emb_array
        .store_chunk(&[0, 0], &embeddings_f16)
        .expect("failed to store embeddings chunk");

    write_children(store, group_path, child_key, children);
}

/// Writes a node whose embeddings are stored as `uint8`, mirroring a build
/// that chose `EmbeddingDtype::UInt8`.
pub fn write_node_uint8(
    store: &Arc<MemoryStore>,
    group_path: &str,
    embeddings: &Array2<f32>,
    child_key: &str,
    children: &Array1<u32>,
) {
    let embeddings_u8 = embeddings.mapv(|x| x as u8);
    let emb_path = format!("{group_path}/embeddings");
    let emb_shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let emb_array = ArrayBuilder::new(emb_shape.clone(), emb_shape, uint8(), 0u8)
        .build(store.clone(), &emb_path)
        .expect("failed to build embeddings array");
    emb_array
        .store_metadata()
        .expect("failed to store embeddings metadata");
    emb_array
        .store_chunk(&[0, 0], &embeddings_u8)
        .expect("failed to store embeddings chunk");

    write_children(store, group_path, child_key, children);
}

/// Like `write_node`, but stores `embeddings` as zarr's `float64` dtype,
/// which `Node::embeddings()` doesn't support (only float16/float32), so it
/// exercises the unsupported-dtype panic path for a dtype `write_node`
/// (always float32) never reaches.
pub fn write_node_unsupported_dtype(
    store: &Arc<MemoryStore>,
    group_path: &str,
    embeddings: &Array2<f32>,
    child_key: &str,
    children: &Array1<u32>,
) {
    let embeddings_f64 = embeddings.mapv(|x| x as f64);
    let emb_path = format!("{group_path}/embeddings");
    let emb_shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let emb_array = ArrayBuilder::new(emb_shape.clone(), emb_shape, float64(), 0.0f64)
        .build(store.clone(), &emb_path)
        .expect("failed to build embeddings array");
    emb_array
        .store_metadata()
        .expect("failed to store embeddings metadata");
    emb_array
        .store_chunk(&[0, 0], &embeddings_f64)
        .expect("failed to store embeddings chunk");

    write_children(store, group_path, child_key, children);
}

/// Writes `info/levels`, `info/metric`, and `info/is_normalized` as rank-0
/// (scalar) arrays, mirroring the fields `ECPBuilder.write_index_info` puts
/// at the root of a real index.
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

/// Writes `info/{name}` as a rank-0 (scalar) array, mirroring
/// `write_info_u32`.
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

/// Writes both id fields level with each other, as a fresh build leaves
/// them. Tests that need the allocator ahead of the count (a crash mid-insert)
/// write `next_item_id` themselves afterwards.
pub fn write_item_counts(store: &Arc<MemoryStore>, total_items: u32) {
    write_info_u32(store, "total_items", total_items);
    write_info_u32(store, "next_item_id", total_items);
}

/// Writes `/rep_item_ids`, mirroring the representative id array a real
/// build leaves at the top level.
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

/// Writes `index_root/embeddings`, mirroring what `ECPBuilder.build_tree_fs`
/// writes for the top-level cluster leaders.
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

pub fn new_memory_store() -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new())
}

pub fn as_readable_listable(store: &Arc<MemoryStore>) -> ReadableListableStorage {
    store.clone()
}

pub fn as_readable_writable_listable(store: &Arc<MemoryStore>) -> ReadableWritableListableStorage {
    store.clone()
}
