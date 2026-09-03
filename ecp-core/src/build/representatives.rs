use ndarray::{Array1, Array2, Axis};
use rand::seq::index::sample;
use zarrs::storage::ReadableWritableListableStorage;

use crate::build::source::EmbeddingsSource;
use crate::build::writer::zarrs_append;

/// How to pick which items become cluster leaders. `"custom"` (the caller
/// already has their own leader ids/embeddings) needs no algorithm here -
/// it's handled by the builder directly.
pub enum RepresentativeStrategy {
    Offset,
    Random,
}

/// Picks `total_clusters` leader ids out of `0..total_items`, sorted
/// ascending (needed by `collect_representatives`'s per-chunk membership
/// check).
pub fn select_representative_ids(
    total_items: usize,
    target_cluster_items: usize,
    strategy: RepresentativeStrategy,
) -> Array1<u32> {
    match strategy {
        RepresentativeStrategy::Offset => {
            (0..total_items as u32).step_by(target_cluster_items).collect()
        }
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
    InMemory { embeddings: Array2<f32>, ids: Array1<u32> },
    PersistedOnly,
}

/// Streams `source` one on-disk chunk at a time, skipping any chunk that
/// contains no `selected_ids`, and persists whichever rows match to
/// `rep_embeddings`/`rep_item_ids`. Also keeps the result in memory for
/// immediate reuse while building the tree, but only if it fits within
/// `memory_limit_bytes`.
///
/// `selected_ids` must already be sorted ascending.
pub fn collect_representatives(
    store: &ReadableWritableListableStorage,
    source: &EmbeddingsSource,
    selected_ids: &Array1<u32>,
    fallback_batch_rows: usize,
    memory_limit_bytes: usize,
    chunk_shape: &[u64],
) -> Representatives {
    let (total_items, dim) = source.shape();
    let batch_rows = source.natural_batch_rows(fallback_batch_rows);
    let keep_in_memory = fits_in_memory(selected_ids.len(), dim, memory_limit_bytes);
    let selected: Vec<u32> = selected_ids.to_vec();

    let mut kept_embeddings: Vec<f32> = Vec::new();
    let mut kept_ids: Vec<u32> = Vec::new();

    let mut start = 0;
    while start < total_items {
        let end = (start + batch_rows).min(total_items);

        let first = selected.partition_point(|&id| (id as usize) < start);
        let in_range = &selected[first..];
        let matched_ids: Vec<u32> = in_range.iter().take_while(|&&id| (id as usize) < end).copied().collect();
        if matched_ids.is_empty() {
            start = end;
            continue;
        }

        let batch = source.read_rows(start, end);
        let matched_rows: Vec<usize> = matched_ids.iter().map(|&id| id as usize - start).collect();
        let matched_embeddings = batch.select(Axis(0), &matched_rows);
        let matched_ids_array = Array1::from_vec(matched_ids.clone());
        zarrs_append(store, "/rep_embeddings", "/rep_item_ids", &matched_embeddings, &matched_ids_array, chunk_shape);

        if keep_in_memory {
            kept_embeddings.extend(matched_embeddings.iter().copied());
            kept_ids.extend(matched_ids);
        }

        start = end;
    }

    if keep_in_memory {
        let count = kept_ids.len();
        Representatives::InMemory {
            embeddings: Array2::from_shape_vec((count, dim), kept_embeddings)
                .expect("collected embeddings didn't match the expected row count"),
            ids: Array1::from_vec(kept_ids),
        }
    } else {
        Representatives::PersistedOnly
    }
}

#[cfg(test)]
#[path = "utests/representatives.rs"]
mod tests;
