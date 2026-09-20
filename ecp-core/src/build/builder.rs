use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2, s};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

use crate::build::representatives::{
    RepresentativeStrategy, Representatives, collect_representatives, fits_in_memory,
    select_representative_ids,
};
use crate::build::source::EmbeddingsSource;
use crate::build::tree::{BuildTreeArgs, build_tree};
use crate::build::writer::{write_index_info, write_index_root, write_info_u32, zarrs_append};
use crate::dtype::EmbeddingDtype;
use crate::metric::Metric;

/// Picks the dtype the index stores embeddings as, which is `embedding_dtype`
/// if the caller set one, else `native`, the source file's own dtype. Warns
/// when `embedding_dtype` can't hold every value `native` can.
fn resolve_dtype(
    embedding_dtype: Option<EmbeddingDtype>,
    native: EmbeddingDtype,
) -> EmbeddingDtype {
    let resolved = embedding_dtype.unwrap_or(native);
    if resolved.narrows(native) {
        log::warn!(
            "writing embeddings as {resolved:?} narrows the source's {native:?} values and loses information; \
             integer targets additionally clamp out-of-range values and truncate fractions"
        );
    }
    resolved
}

/// Default upper limit on one on-disk chunk, in bytes. `calculate_chunk_size`
/// turns it into vectors per chunk.
pub const DEFAULT_MAX_CHUNK_BYTES: usize = 50 * 1024 * 1024;

/// Share of the memory limit a build spends on each batch of vectors and on
/// its node cache. The rest is left for memory this crate doesn't track.
pub(crate) const TRACKED_MEMORY_FRACTION: f64 = 0.8;

/// Picks the chunk size for the index's arrays. Returns how many f32 vectors
/// of length `dim` fit in a chunk of at most `max_chunk_bytes`. Panics if not
/// even one does.
pub fn calculate_chunk_size(dim: usize, max_chunk_bytes: usize) -> u64 {
    let bytes_per_vec = dim * size_of::<f32>();
    assert!(
        bytes_per_vec <= max_chunk_bytes,
        "dim {dim} doesn't fit a single vec in {max_chunk_bytes} bytes"
    );
    (max_chunk_bytes / bytes_per_vec) as u64
}

/// Picks the tree's fan-out, the number of children each node (the root
/// included) gets, so the bottom level has room for every cluster.
/// Fan-out = `total_clusters^(1/levels)`, rounded up.
fn node_size_for(total_clusters: usize, levels: u32) -> usize {
    (total_clusters as f64).powf(1.0 / levels as f64).ceil() as usize
}

/// Builds one index in three steps: `create`, then `select_representatives`
/// or `select_representatives_custom`, then `build`. One build per `Builder`.
pub struct Builder {
    store: ReadableWritableListableStorage,
    levels: u32,
    metric: Metric,
    is_normalized: bool,
    memory_limit_bytes: usize,
    embedding_dtype: Option<EmbeddingDtype>,
    max_chunk_bytes: usize,

    // Set by either select_representatives method
    chunk_shape: Vec<u64>,
    representatives: Option<Representatives>,
    node_size: usize,
    resolved_dtype: EmbeddingDtype,
}

impl Builder {
    /// Starts a build in `store` instead of a path. Otherwise the same as [`Self::create`].
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

    /// Creates an index at `index_path`. `levels` is the number of node levels
    /// below the root. Set `is_normalized` only if every embedding is unit-length,
    /// and leave `embedding_dtype` as `None` to keep each source's own dtype.
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
        Self::new(
            store,
            levels,
            metric,
            is_normalized,
            memory_limit_bytes,
            embedding_dtype,
            max_chunk_bytes,
        )
    }

    /// Picks the representatives the tree is built from, chosen by
    /// `strategy`. `target_cluster_items` is the average cluster size it aims for, and
    /// `fallback_batch_vecs` is the chunk size assumed when `source` isn't chunked.
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

    /// Uses caller-chosen representatives instead of picking them, such as ones
    /// from an external clustering step. `ids[i]` is the id of row `i` of `embeddings`.
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

    /// Builds the tree from `dataset`, whose rows get item ids 0, 1, 2, ...
    /// in order. `fallback_batch_vecs` works as in `select_representatives`.
    pub fn build(&mut self, dataset: &EmbeddingsSource, fallback_batch_vecs: usize) {
        log::info!(
            "building tree: {} levels, metric={:?}",
            self.levels,
            self.metric
        );
        // Ids 0..total_items, so both counters start at total_items
        let (total_items, _) = dataset.shape();
        write_info_u32(&self.store, "total_items", total_items as u32);
        write_info_u32(&self.store, "next_item_id", total_items as u32);

        let representatives = self
            .representatives
            .take()
            .expect("call select_representatives before build");

        // Root holds the first node_size representatives
        let (root_embeddings, representatives_source) = match representatives {
            Representatives::InMemory { embeddings, .. } => {
                let root = embeddings.slice(s![..self.node_size, ..]).to_owned();
                (root, EmbeddingsSource::Memory(embeddings))
            }
            Representatives::PersistedOnly => {
                let source = EmbeddingsSource::from_zarr(
                    self.store.clone().readable_listable(),
                    "/rep_embeddings".to_string(),
                );
                let root = source.read_vecs(0, self.node_size);
                (root, source)
            }
        };

        // Write root, then every level below it
        write_index_root(
            &self.store,
            &root_embeddings,
            &self.chunk_shape,
            self.resolved_dtype,
        );
        build_tree(&BuildTreeArgs {
            store: &self.store,
            root_embeddings: &root_embeddings,
            representatives: &representatives_source,
            dataset,
            total_levels: self.levels,
            metric: self.metric,
            is_normalized: self.is_normalized,
            fallback_batch_vecs,
            chunk_shape: &self.chunk_shape,
            embedding_dtype: self.resolved_dtype,
            memory_limit_bytes: self.memory_limit_bytes,
        });
    }
}

#[cfg(test)]
#[path = "utests/builder.rs"]
mod tests;
