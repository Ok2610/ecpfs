use ndarray::{Array1, Array2, Axis};
use rand::seq::index::sample;
use zarrs::storage::ReadableWritableListableStorage;

use crate::build::builder::TRACKED_MEMORY_FRACTION;
use crate::build::source::EmbeddingsSource;
use crate::build::writer::zarrs_append;
use crate::dtype::EmbeddingDtype;

/// How to pick which items become cluster leaders.
#[derive(Debug, Clone, Copy)]
pub enum RepresentativeStrategy {
    Offset,
    Random,
}

/// Picks leader ids out of `0..total_items`, sorted ascending (needed by
/// `collect_representatives`'s per-chunk membership check). Returns
/// `total_items` divided by `target_cluster_items`, rounded up.
pub fn select_representative_ids(
    total_items: usize,
    target_cluster_items: usize,
    strategy: RepresentativeStrategy,
) -> Array1<u32> {
    match strategy {
        RepresentativeStrategy::Offset => (0..total_items as u32)
            .step_by(target_cluster_items)
            .collect(),
        RepresentativeStrategy::Random => {
            let total_clusters = total_items.div_ceil(target_cluster_items);
            let mut ids: Vec<u32> = sample(&mut rand::rng(), total_items, total_clusters)
                .into_iter()
                .map(|i| i as u32)
                .collect();
            ids.sort_unstable();
            Array1::from_vec(ids)
        }
    }
}

/// Whether `count` embeddings of `dim` floats each fit within
/// `memory_limit_bytes`.
pub fn fits_in_memory(count: usize, dim: usize, memory_limit_bytes: usize) -> bool {
    count.saturating_mul(dim).saturating_mul(size_of::<f32>()) <= memory_limit_bytes
}

/// Whether the representative set stayed in memory (small enough to skip
/// re-reading from disk during tree-building) or was persisted only.
pub enum Representatives {
    InMemory {
        embeddings: Array2<f32>,
        ids: Array1<u32>,
    },
    PersistedOnly,
}

/// Streams `source` in memory-budgeted batches (each at least one full
/// on-disk chunk), skipping any batch that contains no `selected_ids`,
/// and persists whichever vecs match to `rep_embeddings`/`rep_item_ids`.
/// `build_tree` reads them back from disk itself, once per non-leaf pass;
/// this never keeps a copy in memory.
///
/// `selected_ids` must already be sorted ascending.
pub fn collect_representatives(
    store: &ReadableWritableListableStorage,
    source: &EmbeddingsSource,
    selected_ids: &Array1<u32>,
    fallback_batch_vecs: usize,
    memory_limit_bytes: usize,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) {
    let (total_items, dim) = source.shape();
    let tracked_budget = (memory_limit_bytes as f64 * TRACKED_MEMORY_FRACTION) as usize;
    let bytes_per_vec = (dim * size_of::<f32>()).max(1);
    let memory_floor_vecs = (tracked_budget / bytes_per_vec).max(1);
    let batch_vecs = source.chunk_aligned_batch_vecs(memory_floor_vecs, fallback_batch_vecs);
    let selected: Vec<u32> = selected_ids.to_vec();

    let mut start = 0;
    while start < total_items {
        let end = (start + batch_vecs).min(total_items);

        let first = selected.partition_point(|&id| (id as usize) < start);
        let in_range = &selected[first..];
        let matched_ids: Vec<u32> = in_range
            .iter()
            .take_while(|&&id| (id as usize) < end)
            .copied()
            .collect();
        if matched_ids.is_empty() {
            start = end;
            continue;
        }
        log::debug!(
            "processing batch vecs {start}..{end} ({} matched representatives)",
            matched_ids.len()
        );

        let batch = source.read_vecs(start, end);
        let matched_vec_indices: Vec<usize> =
            matched_ids.iter().map(|&id| id as usize - start).collect();
        let matched_embeddings = batch.select(Axis(0), &matched_vec_indices);
        let matched_ids_array = Array1::from_vec(matched_ids);
        zarrs_append(
            store,
            "/rep_embeddings",
            "/rep_item_ids",
            &matched_embeddings,
            &matched_ids_array,
            chunk_shape,
            dtype,
        );

        start = end;
    }
}

#[cfg(test)]
#[path = "utests/representatives.rs"]
mod tests;
