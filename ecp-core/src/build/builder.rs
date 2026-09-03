use std::path::Path;
use std::sync::Arc;

use ndarray::{s, Array1, Array2};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

use crate::build::representatives::{
    collect_representatives, fits_in_memory, select_representative_ids, RepresentativeStrategy, Representatives,
};
use crate::build::source::EmbeddingsSource;
use crate::build::tree::{build_tree, write_index_info, write_index_root};
use crate::build::writer::zarrs_append;
use crate::utils::Metric;

/// Zarr chunks default to this many bytes, matched against `dim` to pick a
/// row count - large enough for good I/O throughput, independent of how
/// many rows any given node ends up with.
pub const DEFAULT_MAX_CHUNK_BYTES: usize = 50 * 1024 * 1024;

/// Max rows that keep one chunk under `max_chunk_bytes`, given `dim` f32 columns.
pub fn calculate_chunk_size(dim: usize, max_chunk_bytes: usize) -> u64 {
    let bytes_per_row = dim * size_of::<f32>();
    assert!(bytes_per_row <= max_chunk_bytes, "dim {dim} doesn't fit a single row in {max_chunk_bytes} bytes");
    (max_chunk_bytes / bytes_per_row) as u64
}

/// How many leaders one node holds, so the tree's fan-out roughly balances
/// across `levels`.
fn node_size_for(total_clusters: usize, levels: u32) -> usize {
    (total_clusters as f64).powf(1.0 / levels as f64).ceil() as usize
}

/// Orchestrates one index build: pick representatives, then descend the
/// full tree from them. Owns the store it writes to.
pub struct Builder {
    store: ReadableWritableListableStorage,
    levels: u32,
    metric: Metric,
    is_normalized: bool,
    memory_limit_bytes: usize,
    chunk_shape: Vec<u64>,
    representatives: Option<Representatives>,
    node_size: usize,
}

impl Builder {
    /// Writes `info/*` immediately, before any representatives exist.
    pub fn new(
        store: ReadableWritableListableStorage,
        levels: u32,
        metric: Metric,
        is_normalized: bool,
        memory_limit_bytes: usize,
    ) -> Self {
        write_index_info(&store, levels, metric, is_normalized);
        Builder {
            store,
            levels,
            metric,
            is_normalized,
            memory_limit_bytes,
            chunk_shape: Vec::new(),
            representatives: None,
            node_size: 0,
        }
    }

    /// Creates a fresh `FilesystemStore` at `index_path` and builds into it.
    pub fn create(index_path: &Path, levels: u32, metric: Metric, is_normalized: bool, memory_limit_bytes: usize) -> Self {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(index_path).expect("Failed to create store"));
        Self::new(store, levels, metric, is_normalized, memory_limit_bytes)
    }

    /// Picks leaders out of `source` via `strategy` and persists them to
    /// `/rep_embeddings`/`/rep_item_ids`. Must be called (or
    /// `select_representatives_custom`) before `build`.
    pub fn select_representatives(
        &mut self,
        source: &EmbeddingsSource,
        target_cluster_items: usize,
        strategy: RepresentativeStrategy,
        fallback_batch_rows: usize,
    ) {
        let (total_items, dim) = source.shape();
        self.chunk_shape = vec![calculate_chunk_size(dim, DEFAULT_MAX_CHUNK_BYTES), dim as u64];

        let selected_ids = select_representative_ids(total_items, target_cluster_items, strategy);
        self.node_size = node_size_for(selected_ids.len(), self.levels);
        self.representatives = Some(collect_representatives(
            &self.store,
            source,
            &selected_ids,
            fallback_batch_rows,
            self.memory_limit_bytes,
            &self.chunk_shape,
        ));
    }

    /// Uses caller-supplied leaders directly instead of running a
    /// selection strategy - e.g. representatives chosen by an external
    /// clustering step.
    pub fn select_representatives_custom(&mut self, ids: Array1<u32>, embeddings: Array2<f32>) {
        let dim = embeddings.ncols();
        self.chunk_shape = vec![calculate_chunk_size(dim, DEFAULT_MAX_CHUNK_BYTES), dim as u64];
        zarrs_append(&self.store, "/rep_embeddings", "/rep_item_ids", &embeddings, &ids, &self.chunk_shape);

        self.node_size = node_size_for(ids.len(), self.levels);
        self.representatives = Some(if fits_in_memory(ids.len(), dim, self.memory_limit_bytes) {
            Representatives::InMemory { embeddings, ids }
        } else {
            Representatives::PersistedOnly
        });
    }

    /// Writes `index_root` and descends the full tree over `dataset`.
    pub fn build(&mut self, dataset: &EmbeddingsSource, fallback_batch_rows: usize) {
        let representatives =
            self.representatives.take().expect("call select_representatives before build");

        let (root_embeddings, representatives_source) = match representatives {
            Representatives::InMemory { embeddings, .. } => {
                let root = embeddings.slice(s![..self.node_size, ..]).to_owned();
                (root, EmbeddingsSource::Memory(embeddings))
            }
            Representatives::PersistedOnly => {
                let source = EmbeddingsSource::from_zarr(self.store.clone().readable_listable(), "/rep_embeddings".to_string());
                let root = source.read_rows(0, self.node_size);
                (root, source)
            }
        };

        write_index_root(&self.store, &root_embeddings, &self.chunk_shape);
        build_tree(
            &self.store,
            &root_embeddings,
            &representatives_source,
            dataset,
            self.levels,
            self.metric,
            self.is_normalized,
            fallback_batch_rows,
            &self.chunk_shape,
        );
    }
}

#[cfg(test)]
#[path = "utests/builder.rs"]
mod tests;
