//! Proves `Builder` agrees with search end to end, the same way
//! `tests/build_tree.rs` proves it for the hand-assembled `build_tree`
//! path: build via `Builder`, then load and search with `Index::load`.

use ndarray::array;
use std::collections::HashSet;
use std::sync::Arc;
use zarrs::array::data_type::float32;
use zarrs::array::ArrayBuilder;
use zarrs::filesystem::FilesystemStore;

use ecp_core::build::builder::Builder;
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
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

/// Same two-well-separated-clusters-of-4 geometry `tests/build_tree.rs`
/// uses, but selected and built entirely through `Builder`.
#[test]
fn builder_produces_a_structure_that_searches_correctly() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings(
        &store,
        "/dataset",
        &array![
            [0.0f32, 0.0], [0.4, 0.4], [1.0, 1.0], [1.4, 1.4],
            [10.0, 10.0], [10.4, 10.4], [11.0, 11.0], [11.4, 11.4]
        ],
    );
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::new(store.clone(), 2, Metric::L2, false, 1_000_000_000);
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    let mut index = Index::load(index_path);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}
