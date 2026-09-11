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
use crate::utils::{EmbeddingDtype, Metric};

/// `requested` if set, else `native`. Warns on an `F32`-to-`F16` downcast.
fn resolve_dtype(requested: Option<EmbeddingDtype>, native: EmbeddingDtype) -> EmbeddingDtype {
    let resolved = requested.unwrap_or(native);
    if resolved == EmbeddingDtype::F16 && native == EmbeddingDtype::F32 {
        log::warn!("writing embeddings as f16 downcasts the source's f32 precision");
    }
    resolved
}

/// Zarr chunks default to this many bytes, matched against `dim` to pick a
/// vec count
pub const DEFAULT_MAX_CHUNK_BYTES: usize = 50 * 1024 * 1024;

/// Fraction of `memory_limit_bytes` actually budgeted for a build pass's
/// batch buffer (plus, in `build_tree`, its node cache); the rest is
/// headroom for everything else the process holds, untracked here.
pub(crate) const TRACKED_MEMORY_FRACTION: f64 = 0.8;

/// Max vecs that keep one chunk under `max_chunk_bytes`, given `dim` f32 columns.
pub fn calculate_chunk_size(dim: usize, max_chunk_bytes: usize) -> u64 {
    let bytes_per_vec = dim * size_of::<f32>();
    assert!(bytes_per_vec <= max_chunk_bytes, "dim {dim} doesn't fit a single vec in {max_chunk_bytes} bytes");
    (max_chunk_bytes / bytes_per_vec) as u64
}

/// The per-node fan-out `ns`: root holds `ns` leaders, and each level's
/// nodes hold `ns` children (see `build_tree`'s docs for how this composes).
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
    embedding_dtype: Option<EmbeddingDtype>,
    max_chunk_bytes: usize,
    chunk_shape: Vec<u64>,
    representatives: Option<Representatives>,
    node_size: usize,
    /// Set once by `select_representatives`/`select_representatives_custom`.
    resolved_dtype: EmbeddingDtype,
}

impl Builder {
    /// Writes `info/*` immediately, before any representatives exist.
    /// `embedding_dtype` of `None` matches each source's own dtype.
    pub fn new(
        store: ReadableWritableListableStorage,
        levels: u32,
        metric: Metric,
        is_normalized: bool,
        memory_limit_bytes: usize,
        embedding_dtype: Option<EmbeddingDtype>,
        max_chunk_bytes: usize,
    ) -> Self {
        write_index_info(&store, levels, metric, is_normalized);
        Builder {
            store,
            levels,
            metric,
            is_normalized,
            memory_limit_bytes,
            embedding_dtype,
            max_chunk_bytes,
            chunk_shape: Vec::new(),
            representatives: None,
            node_size: 0,
            resolved_dtype: EmbeddingDtype::F32,
        }
    }

    /// Creates a fresh `FilesystemStore` at `index_path` and builds into it.
    pub fn create(
        index_path: &Path,
        levels: u32,
        metric: Metric,
        is_normalized: bool,
        memory_limit_bytes: usize,
        embedding_dtype: Option<EmbeddingDtype>,
        max_chunk_bytes: usize,
    ) -> Self {
        log::info!("creating index at {}", index_path.display());
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(index_path).expect("Failed to create store"));
        Self::new(store, levels, metric, is_normalized, memory_limit_bytes, embedding_dtype, max_chunk_bytes)
    }

    /// Picks leaders out of `source` via `strategy` and persists them to
    /// `/rep_embeddings`/`/rep_item_ids`. Must be called (or
    /// `select_representatives_custom`) before `build`.
    pub fn select_representatives(
        &mut self,
        source: &EmbeddingsSource,
        target_cluster_items: usize,
        strategy: RepresentativeStrategy,
        fallback_batch_vecs: usize,
    ) {
        let (total_items, dim) = source.shape();
        self.chunk_shape = vec![calculate_chunk_size(dim, self.max_chunk_bytes), dim as u64];
        self.resolved_dtype = resolve_dtype(self.embedding_dtype, source.native_dtype());

        let selected_ids = select_representative_ids(total_items, target_cluster_items, strategy);
        log::info!(
            "selected {} representatives via {strategy:?} from {total_items} items (target_cluster_items={target_cluster_items})",
            selected_ids.len()
        );
        self.node_size = node_size_for(selected_ids.len(), self.levels);
        collect_representatives(
            &self.store,
            source,
            &selected_ids,
            fallback_batch_vecs,
            self.memory_limit_bytes,
            &self.chunk_shape,
            self.resolved_dtype,
        );
        self.representatives = Some(Representatives::PersistedOnly);
    }

    /// Uses caller-supplied leaders directly instead of running a
    /// selection strategy, for representatives chosen by an external
    /// clustering step.
    pub fn select_representatives_custom(&mut self, ids: Array1<u32>, embeddings: Array2<f32>) {
        assert_eq!(
            ids.len(),
            embeddings.nrows(),
            "ids and embeddings must have the same length ({} ids, {} embeddings rows)",
            ids.len(),
            embeddings.nrows()
        );
        let dim = embeddings.ncols();
        self.chunk_shape = vec![calculate_chunk_size(dim, self.max_chunk_bytes), dim as u64];
        self.resolved_dtype = resolve_dtype(self.embedding_dtype, EmbeddingDtype::F32);
        zarrs_append(
            &self.store,
            "/rep_embeddings",
            "/rep_item_ids",
            &embeddings,
            &ids,
            &self.chunk_shape,
            self.resolved_dtype,
        );

        self.node_size = node_size_for(ids.len(), self.levels);
        self.representatives = Some(if fits_in_memory(ids.len(), dim, self.memory_limit_bytes) {
            Representatives::InMemory { embeddings, ids }
        } else {
            Representatives::PersistedOnly
        });
    }

    /// Writes `index_root` and descends the full tree over `dataset`.
    pub fn build(&mut self, dataset: &EmbeddingsSource, fallback_batch_vecs: usize) {
        log::info!("building tree: {} levels, metric={:?}", self.levels, self.metric);
        let representatives =
            self.representatives.take().expect("call select_representatives before build");

        let (root_embeddings, representatives_source) = match representatives {
            Representatives::InMemory { embeddings, .. } => {
                let root = embeddings.slice(s![..self.node_size, ..]).to_owned();
                (root, EmbeddingsSource::Memory(embeddings))
            }
            Representatives::PersistedOnly => {
                let source = EmbeddingsSource::from_zarr(self.store.clone().readable_listable(), "/rep_embeddings".to_string());
                let root = source.read_vecs(0, self.node_size);
                (root, source)
            }
        };

        write_index_root(&self.store, &root_embeddings, &self.chunk_shape, self.resolved_dtype);
        build_tree(
            &self.store,
            &root_embeddings,
            &representatives_source,
            dataset,
            self.levels,
            self.metric,
            self.is_normalized,
            fallback_batch_vecs,
            &self.chunk_shape,
            self.resolved_dtype,
            self.memory_limit_bytes,
        );
    }
}

#[cfg(test)]
#[path = "utests/builder.rs"]
mod tests;
