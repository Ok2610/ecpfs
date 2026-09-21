use ndarray::{Array1, Array2, Axis};
use rand::seq::index::sample;
use zarrs::storage::ReadableWritableListableStorage;

use crate::build::builder::TRACKED_MEMORY_FRACTION;
use crate::build::source::EmbeddingsSource;
use crate::build::writer::zarrs_append;
use crate::dtype::EmbeddingDtype;
use crate::error::Result;

/// How `select_representatives` picks representatives, the items that every
/// other item is clustered around.
#[derive(Debug, Clone, Copy)]
pub enum RepresentativeStrategy {
    /// Items spaced `target_cluster_items` apart, starting at item 0.
    Offset,
    /// A uniform random sample, as many as `Offset` picks.
    Random,
}

/// Picks which items become representatives and returns their ids, sorted ascending.
/// Number of representatives = `total_items / target_cluster_items`, rounded up.
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

/// Checks whether `count` f32 embeddings of length `dim` fit under the memory limit.
pub fn fits_in_memory(count: usize, dim: usize, memory_limit_bytes: usize) -> bool {
    count.saturating_mul(dim).saturating_mul(size_of::<f32>()) <= memory_limit_bytes
}

/// Whether `build` reads the representatives from memory, when they were
/// small enough to keep, or from disk.
pub enum Representatives {
    InMemory {
        embeddings: Array2<f32>,
        ids: Array1<u32>,
    },
    PersistedOnly,
}

/// Saves the chosen representatives into the index. Copies the rows of
/// `source` listed in `selected_ids`, which must be sorted ascending. Reads
/// whole chunks and skips any chunk with no selected row.
pub fn collect_representatives(
    store: &ReadableWritableListableStorage,
    source: &EmbeddingsSource,
    selected_ids: &Array1<u32>,
    fallback_batch_vecs: usize,
    memory_limit_bytes: usize,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) -> Result<()> {
    let (total_items, dim) = source.shape()?;
    let tracked_budget = (memory_limit_bytes as f64 * TRACKED_MEMORY_FRACTION) as usize;
    let bytes_per_vec = (dim * size_of::<f32>()).max(1);
    let memory_floor_vecs = (tracked_budget / bytes_per_vec).max(1);
    let batch_vecs = source.chunk_aligned_batch_vecs(memory_floor_vecs, fallback_batch_vecs)?;
    let selected: Vec<u32> = selected_ids.to_vec();

    let mut start = 0;
    while start < total_items {
        let end = (start + batch_vecs).min(total_items);

        // Selected ids in this batch
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

        // Read the batch and append the selected rows
        let batch = source.read_vecs(start, end)?;
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
        )?;

        start = end;
    }
    Ok(())
}

#[cfg(test)]
#[path = "utests/representatives.rs"]
mod tests;
