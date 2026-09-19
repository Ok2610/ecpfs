//! Helpers shared by the integration tests. Each file in `tests/` compiles
//! as its own crate, so shared code lives here and each file declares
//! `mod common;`.

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array2, array};
use tempfile::TempDir;
use zarrs::array::ArrayBuilder;
use zarrs::array::data_type::float32;
use zarrs::filesystem::FilesystemStore;

use ecp_core::build::builder::{Builder, DEFAULT_MAX_CHUNK_BYTES};
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::utils::{EmbeddingDtype, Metric};

/// Writes `embeddings` to `path` as a single-chunk float32 array.
pub fn write_embeddings(store: &Arc<FilesystemStore>, path: &str, embeddings: &Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, float32(), 0.0f32)
        .build(store.clone(), path)
        .expect("failed to build embeddings array");
    array
        .store_metadata()
        .expect("failed to store embeddings metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[
                0..embeddings.nrows() as u64,
                0..embeddings.ncols() as u64,
            ]),
            embeddings,
        )
        .expect("failed to store embeddings");
}

/// Two well-separated clusters of 4 items each: items 0-3 near the origin,
/// items 4-7 near (10, 10).
pub fn two_clusters() -> Array2<f32> {
    array![
        [0.0f32, 0.0],
        [0.4, 0.4],
        [1.0, 1.0],
        [1.4, 1.4],
        [10.0, 10.0],
        [10.4, 10.4],
        [11.0, 11.0],
        [11.4, 11.4]
    ]
}

/// Builds a 2-level L2 index over `vectors` in a fresh temp directory.
/// The directory is deleted when the returned `TempDir` is dropped, so hold
/// it for as long as the index is used.
pub fn build_index(vectors: &Array2<f32>, dtype: Option<EmbeddingDtype>) -> (TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));
    write_embeddings(&store, "/dataset", vectors);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(
        &index_path,
        2,
        Metric::L2,
        false,
        1_000_000_000,
        dtype,
        DEFAULT_MAX_CHUNK_BYTES,
    );
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    (tmp, index_path)
}
