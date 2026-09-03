//! Without a compressor, a zarr chunk is padded to its full declared size
//! on disk regardless of how much is actually written. Since chunk_shape
//! here is sized for I/O throughput (tens of MB), not expected cluster
//! size (eCP doesn't enforce target_cluster_items - a single cluster can
//! absorb a wildly disproportionate share of the dataset), most nodes are
//! mostly-empty relative to their chunk. This proves compression actually
//! keeps that cheap, on a real filesystem where padding would otherwise
//! show up as real disk usage.

use ndarray::{array, Array1, Array2};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;
use std::sync::Arc;

use ecp_core::build::writer::zarrs_append;

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path).expect("failed to read dir") {
        let entry = entry.expect("failed to read dir entry");
        let meta = entry.metadata().expect("failed to read metadata");
        total += if meta.is_dir() { dir_size(&entry.path()) } else { meta.len() };
    }
    total
}

#[test]
fn a_mostly_empty_node_stays_small_on_disk_despite_a_large_chunk_shape() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(tmp.path()).expect("failed to create filesystem store"));

    // A chunk sized for ~50MB at dim=768 (~17,000 rows), but this node
    // only ever gets 3 real rows - the common case, not the pathological
    // millions-of-items one, but the same mostly-empty-chunk shape.
    let dim = 768;
    let chunk_shape = [17_000u64, dim as u64];
    let embeddings = Array2::from_elem((3, dim), 1.0f32);
    let ids: Array1<u32> = array![1, 2, 3];

    zarrs_append(&store, "/node/embeddings", "/node/item_ids", &embeddings, &ids, &chunk_shape);

    let actual_size = dir_size(&tmp.path().join("node"));
    let naive_padded_size = chunk_shape[0] * chunk_shape[1] * 4;
    assert!(
        actual_size < naive_padded_size / 100,
        "expected compression to keep a 3-row node far below its {naive_padded_size}-byte padded size, got {actual_size} bytes"
    );
}
