#![cfg(test)]

use std::sync::Arc;

use half::f16;
use ndarray::{Array1, Array2};
use zarrs::array::data_type::{bool, float16, float32, float64, string, uint32};
use zarrs::array::{ArrayBuilder, FillValueMetadata};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

fn write_children(store: &Arc<MemoryStore>, group_path: &str, child_key: &str, children: &Array1<u32>) {
    let child_path = format!("{group_path}/{child_key}");
    let child_shape = vec![children.len() as u64];
    let child_array = 
        ArrayBuilder::new(child_shape.clone(), child_shape, uint32(), 0u32)
        .build(store.clone(), &child_path)
        .expect("failed to build children array");
    child_array.store_metadata().expect("failed to store children metadata");
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
    let emb_array = 
        ArrayBuilder::new(emb_shape.clone(), emb_shape, float32(), 0.0f32)
        .build(store.clone(), &emb_path)
        .expect("failed to build embeddings array");
    emb_array.store_metadata().expect("failed to store embeddings metadata");
    emb_array
        .store_chunk(&[0, 0], embeddings)
        .expect("failed to store embeddings chunk");

    write_children(store, group_path, child_key, children);
}

/// Like `write_node`, but stores `embeddings` as zarr's `float16` dtype - exercises
/// `Node::embeddings()`'s f16-upcast branch, which `write_node` (always float32)
/// never reaches.
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
    let emb_array = 
        ArrayBuilder::new(emb_shape.clone(), emb_shape, float16(), f16::from_f32(0.0))
        .build(store.clone(), &emb_path)
        .expect("failed to build embeddings array");
    emb_array.store_metadata().expect("failed to store embeddings metadata");
    emb_array
        .store_chunk(&[0, 0], &embeddings_f16)
        .expect("failed to store embeddings chunk");

    write_children(store, group_path, child_key, children);
}

/// Like `write_node`, but stores `embeddings` as zarr's `float64` dtype, which
/// `Node::embeddings()` doesn't support (only float16/float32) - exercises the
/// "unknown datatype" panic path for a dtype `write_node` (always float32) never
/// reaches.
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
    let emb_array = 
        ArrayBuilder::new(emb_shape.clone(), emb_shape, float64(), 0.0f64)
        .build(store.clone(), &emb_path)
        .expect("failed to build embeddings array");
    emb_array.store_metadata().expect("failed to store embeddings metadata");
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

    let levels_array = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), uint32(), 0u32)
        .build(store.clone(), "/info/levels")
        .expect("failed to build info/levels array");
    levels_array.store_metadata().expect("failed to store info/levels metadata");
    levels_array
        .store_chunk(&[], vec![levels])
        .expect("failed to store info/levels chunk");

    let metric_array = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), string(), "")
        .build(store.clone(), "/info/metric")
        .expect("failed to build info/metric array");
    metric_array.store_metadata().expect("failed to store info/metric metadata");
    metric_array
        .store_chunk(&[], vec![metric.to_string()])
        .expect("failed to store info/metric chunk");

    let is_normalized_array = ArrayBuilder::new(scalar_shape.clone(), scalar_shape, bool(), FillValueMetadata::Bool(false))
        .build(store.clone(), "/info/is_normalized")
        .expect("failed to build info/is_normalized array");
    is_normalized_array.store_metadata().expect("failed to store info/is_normalized metadata");
    is_normalized_array
        .store_chunk(&[], vec![is_normalized])
        .expect("failed to store info/is_normalized chunk");
}

/// Writes `index_root/embeddings`, mirroring what `ECPBuilder.build_tree_fs`
/// writes for the top-level cluster leaders.
pub fn write_index_root(store: &Arc<MemoryStore>, embeddings: &Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let root_array = ArrayBuilder::new(shape.clone(), shape, float32(), 0.0f32)
        .build(store.clone(), "/index_root/embeddings")
        .expect("failed to build index_root/embeddings array");
    root_array.store_metadata().expect("failed to store index_root/embeddings metadata");
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
