//! Proves build and search agree end to end, through the same public
//! entry points a real caller uses: build a tree with `build_tree`, then
//! load and search it with `Index::load`. If `build_tree`'s assignment
//! ever disagreed with search's ranking for the same points, this would
//! catch it - a self-search from the tree's own data wouldn't come back
//! in nearest-to-farthest order.

use ndarray::array;
use std::collections::HashSet;
use std::sync::Arc;
use zarrs::array::data_type::float32;
use zarrs::array::ArrayBuilder;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

use ecp_core::build::source::EmbeddingsSource;
use ecp_core::build::tree::{build_tree, write_index_info, write_index_root};
use ecp_core::search::Index;
use ecp_core::utils::Metric;

fn write_embeddings(store: &Arc<FilesystemStore>, path: &str, embeddings: &ndarray::Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, float32(), 0.0f32)
        .build(store.clone(), path)
        .expect("failed to build embeddings array");
    array.store_metadata().expect("failed to store embeddings metadata");
    array
        .store_array_subset(&zarrs::array::ArraySubset::new_with_ranges(&[0..embeddings.nrows() as u64, 0..embeddings.ncols() as u64]), embeddings)
        .expect("failed to store embeddings");
}

/// Same geometry `search::index::tests::build_test_index` hand-builds: leaders
/// 0-3 = items 0,2,4,6; root = leaders 0-1; two well-separated clusters of
/// 4 items each - but built here via `build_tree` instead of a struct
/// literal.
#[test]
fn build_tree_produces_a_structure_that_searches_correctly() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));
    let store_rw: ReadableWritableListableStorage = store.clone();

    write_embeddings(&store, "/rep_embeddings", &array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]]);
    let representatives = EmbeddingsSource::open(&index_path, "rep_embeddings");

    write_embeddings(
        &store,
        "/dataset",
        &array![
            [0.0f32, 0.0], [0.4, 0.4], [1.0, 1.0], [1.4, 1.4],
            [10.0, 10.0], [10.4, 10.4], [11.0, 11.0], [11.4, 11.4]
        ],
    );
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let root_embeddings = array![[0.0f32, 0.0], [1.0, 1.0]];

    write_index_info(&store_rw, 2, Metric::L2, false);
    write_index_root(&store_rw, &root_embeddings, &[100, 2]);
    build_tree(&store_rw, &root_embeddings, &representatives, &dataset, 2, Metric::L2, false, 100, &[100, 2]);

    let mut index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}
