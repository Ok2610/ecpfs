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
use crate::error::{EcpError, Result, ResultExt};
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

/// Default chunk size for the representative arrays, in bytes.
pub const DEFAULT_REP_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// Default chunk size for the tree nodes, in bytes.
pub const DEFAULT_NODE_CHUNK_BYTES: usize = 512 * 1024;

/// Share of the memory limit a build spends on each batch of vectors and on
/// its node cache. The rest is left for memory this crate doesn't track.
pub(crate) const TRACKED_MEMORY_FRACTION: f64 = 0.8;

/// The chunk sizes a build writes, in bytes.
#[derive(Debug, Clone, Copy)]
pub struct ChunkSizes {
    /// Chunk size for `/rep_embeddings` and `/rep_item_ids`.
    pub rep_chunk_bytes: usize,
    /// Chunk size for the root and every tree node.
    pub node_chunk_bytes: usize,
}

impl Default for ChunkSizes {
    fn default() -> Self {
        ChunkSizes {
            rep_chunk_bytes: DEFAULT_REP_CHUNK_BYTES,
            node_chunk_bytes: DEFAULT_NODE_CHUNK_BYTES,
        }
    }
}

/// Returns how many vectors of length `dim` fit in `chunk_bytes` when stored
/// as `dtype`. Errors for a dimension of 0, or if one vector is wider than
/// `chunk_bytes`.
pub fn chunk_rows(dim: usize, dtype: EmbeddingDtype, chunk_bytes: usize) -> Result<u64> {
    if dim == 0 {
        return Err(EcpError::InvalidInput(
            "the embedding dimension must be at least 1".to_string(),
        ));
    }
    let bytes_per_vec = dim.saturating_mul(dtype.bytes());
    if bytes_per_vec > chunk_bytes {
        return Err(EcpError::InvalidInput(format!(
            "one vector of dimension {dim} stored as {dtype:?} takes {bytes_per_vec} bytes, \
             wider than the {chunk_bytes}-byte chunk"
        )));
    }
    Ok((chunk_bytes / bytes_per_vec) as u64)
}

/// Picks the tree's fan-out, the number of children each node (the root
/// included) gets, so the bottom level has room for every cluster.
/// Fan-out = `total_clusters^(1/levels)`, rounded up.
fn node_size_for(total_clusters: usize, levels: u32) -> Result<usize> {
    if levels == 0 {
        return Err(EcpError::InvalidInput(
            "levels must be at least 1".to_string(),
        ));
    }
    Ok((total_clusters as f64).powf(1.0 / levels as f64).ceil() as usize)
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
    chunks: ChunkSizes,

    // Set by either select_representatives method
    rep_chunk_shape: Vec<u64>,
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
        chunks: ChunkSizes,
    ) -> Result<Self> {
        if levels == 0 {
            return Err(EcpError::InvalidInput(
                "levels must be at least 1".to_string(),
            ));
        }
        write_index_info(&store, levels, metric, is_normalized)?;
        Ok(Builder {
            store,
            levels,
            metric,
            is_normalized,
            memory_limit_bytes,
            embedding_dtype,
            chunks,
            rep_chunk_shape: Vec::new(),
            representatives: None,
            node_size: 0,
            resolved_dtype: EmbeddingDtype::F32,
        })
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
        chunks: ChunkSizes,
    ) -> Result<Self> {
        log::info!("creating index at {}", index_path.display());
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(index_path).store_err("failed to create store")?);
        Self::new(
            store,
            levels,
            metric,
            is_normalized,
            memory_limit_bytes,
            embedding_dtype,
            chunks,
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
    ) -> Result<()> {
        let (total_items, dim) = source.shape()?;
        self.resolved_dtype = resolve_dtype(self.embedding_dtype, source.native_dtype()?);
        self.rep_chunk_shape = vec![
            chunk_rows(dim, self.resolved_dtype, self.chunks.rep_chunk_bytes)?,
            dim as u64,
        ];

        let selected_ids = select_representative_ids(total_items, target_cluster_items, strategy);
        log::info!(
            "selected {} representatives via {strategy:?} from {total_items} items (target_cluster_items={target_cluster_items})",
            selected_ids.len()
        );
        self.node_size = node_size_for(selected_ids.len(), self.levels)?;
        collect_representatives(
            &self.store,
            source,
            &selected_ids,
            fallback_batch_vecs,
            self.memory_limit_bytes,
            &self.rep_chunk_shape,
            self.resolved_dtype,
        )?;
        self.representatives = Some(Representatives::PersistedOnly);
        Ok(())
    }

    /// Uses caller-chosen representatives instead of picking them, such as ones
    /// from an external clustering step. `ids[i]` is the id of row `i` of `embeddings`.
    pub fn select_representatives_custom(
        &mut self,
        ids: Array1<u32>,
        embeddings: Array2<f32>,
    ) -> Result<()> {
        if ids.len() != embeddings.nrows() {
            return Err(EcpError::InvalidInput(format!(
                "ids and embeddings must have the same length ({} ids, {} embeddings rows)",
                ids.len(),
                embeddings.nrows()
            )));
        }
        let dim = embeddings.ncols();
        self.resolved_dtype = resolve_dtype(self.embedding_dtype, EmbeddingDtype::F32);
        self.rep_chunk_shape = vec![
            chunk_rows(dim, self.resolved_dtype, self.chunks.rep_chunk_bytes)?,
            dim as u64,
        ];
        zarrs_append(
            &self.store,
            "/rep_embeddings",
            "/rep_item_ids",
            &embeddings,
            &ids,
            &self.rep_chunk_shape,
            self.resolved_dtype,
        )?;

        self.node_size = node_size_for(ids.len(), self.levels)?;
        self.representatives = Some(if fits_in_memory(ids.len(), dim, self.memory_limit_bytes) {
            Representatives::InMemory { embeddings, ids }
        } else {
            Representatives::PersistedOnly
        });
        Ok(())
    }

    /// Builds the tree from `dataset`, whose rows get item ids 0, 1, 2, ...
    /// in order. `fallback_batch_vecs` works as in `select_representatives`.
    pub fn build(&mut self, dataset: &EmbeddingsSource, fallback_batch_vecs: usize) -> Result<()> {
        log::info!(
            "building tree: {} levels, metric={:?}",
            self.levels,
            self.metric
        );
        // Ids 0..total_items, so both counters start at total_items
        let (total_items, dim) = dataset.shape()?;
        write_info_u32(&self.store, "total_items", total_items as u32)?;
        write_info_u32(&self.store, "next_item_id", total_items as u32)?;

        let node_chunk_shape = vec![
            chunk_rows(dim, self.resolved_dtype, self.chunks.node_chunk_bytes)?,
            dim as u64,
        ];
        log::info!(
            "node arrays chunk by {} vecs of {:?}",
            node_chunk_shape[0],
            self.resolved_dtype
        );

        let representatives = self.representatives.take().ok_or_else(|| {
            EcpError::Usage("call select_representatives before build".to_string())
        })?;

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
                let root = source.read_vecs(0, self.node_size)?;
                (root, source)
            }
        };

        // Write root, then every level below it
        write_index_root(
            &self.store,
            &root_embeddings,
            &node_chunk_shape,
            self.resolved_dtype,
        )?;
        build_tree(&BuildTreeArgs {
            store: &self.store,
            root_embeddings: &root_embeddings,
            representatives: &representatives_source,
            dataset,
            total_levels: self.levels,
            metric: self.metric,
            is_normalized: self.is_normalized,
            fallback_batch_vecs,
            chunk_shape: &node_chunk_shape,
            embedding_dtype: self.resolved_dtype,
            memory_limit_bytes: self.memory_limit_bytes,
        })
    }
}

#[cfg(test)]
#[path = "utests/builder.rs"]
mod tests;
