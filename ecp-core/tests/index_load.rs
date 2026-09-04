//! Exercises `Index::load` against a real on-disk `FilesystemStore`, since
//! every other test in this crate uses an in-memory store instead (see
//! `search::index::tests::load_from_store_*` for the fast, storage-agnostic
//! coverage of the same parsing logic). This is the one place that proves
//! the real filesystem code path - directory listing, path handling - also
//! works, not just the generic zarr-store logic.

use ndarray::array;
use zarrs::array::data_type::{bool, float32, string, uint32};
use zarrs::array::{ArrayBuilder, FillValueMetadata};
use zarrs::filesystem::FilesystemStore;
use std::collections::HashSet;
use std::sync::Arc;

use ecp_core::search::Index;

fn write_scalar_u32(store: &Arc<FilesystemStore>, path: &str, value: u32) {
    let shape: Vec<u64> = vec![];
    let array = ArrayBuilder::new(shape.clone(), shape, uint32(), 0u32)
        .build(store.clone(), path)
        .expect("failed to build scalar u32 array");
    array.store_metadata().expect("failed to store scalar metadata");
    array.store_chunk(&[], vec![value]).expect("failed to store scalar chunk");
}

fn write_scalar_string(store: &Arc<FilesystemStore>, path: &str, value: &str) {
    let shape: Vec<u64> = vec![];
    let array = ArrayBuilder::new(shape.clone(), shape, string(), "")
        .build(store.clone(), path)
        .expect("failed to build scalar string array");
    array.store_metadata().expect("failed to store scalar metadata");
    array
        .store_chunk(&[], vec![value.to_string()])
        .expect("failed to store scalar chunk");
}

fn write_scalar_bool(store: &Arc<FilesystemStore>, path: &str, value: bool) {
    let shape: Vec<u64> = vec![];
    let array = ArrayBuilder::new(shape.clone(), shape, bool(), FillValueMetadata::Bool(false))
        .build(store.clone(), path)
        .expect("failed to build scalar bool array");
    array.store_metadata().expect("failed to store scalar metadata");
    array.store_chunk(&[], vec![value]).expect("failed to store scalar chunk");
}

fn write_node(store: &Arc<FilesystemStore>, group_path: &str, embeddings: &ndarray::Array2<f32>, child_key: &str, children: &ndarray::Array1<u32>) {
    let emb_shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let emb_array = ArrayBuilder::new(emb_shape.clone(), emb_shape, float32(), 0.0f32)
        .build(store.clone(), &format!("{group_path}/embeddings"))
        .expect("failed to build embeddings array");
    emb_array.store_metadata().expect("failed to store embeddings metadata");
    emb_array.store_chunk(&[0, 0], embeddings).expect("failed to store embeddings chunk");

    let child_shape = vec![children.len() as u64];
    let child_array = ArrayBuilder::new(child_shape.clone(), child_shape, uint32(), 0u32)
        .build(store.clone(), &format!("{group_path}/{child_key}"))
        .expect("failed to build children array");
    child_array.store_metadata().expect("failed to store children metadata");
    child_array.store_chunk(&[0], children).expect("failed to store children chunk");
}

#[test]
fn load_reads_a_real_index_from_disk_and_searches_correctly() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");

    let store: Arc<FilesystemStore> =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_scalar_u32(&store, "/info/levels", 1);
    write_scalar_string(&store, "/info/metric", "L2");
    write_scalar_bool(&store, "/info/is_normalized", false);

    let root_shape = vec![2u64, 2];
    let root_array = ArrayBuilder::new(root_shape.clone(), root_shape, float32(), 0.0f32)
        .build(store.clone(), "/index_root/embeddings")
        .expect("failed to build root embeddings array");
    root_array.store_metadata().expect("failed to store root metadata");
    root_array
        .store_chunk(&[0, 0], &array![[0.0f32, 0.0], [10.0, 10.0]])
        .expect("failed to store root chunk");

    write_node(&store, "/lvl_1/node_0", &array![[0.0f32, 0.0], [0.4, 0.4]], "item_ids", &array![0u32, 1]);
    write_node(&store, "/lvl_1/node_1", &array![[10.0f32, 10.0], [10.4, 10.4]], "item_ids", &array![2u32, 3]);

    let mut index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}
