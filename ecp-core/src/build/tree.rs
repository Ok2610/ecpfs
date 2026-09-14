use std::sync::{Arc, RwLock};

use dashmap::DashMap;
use moka::sync::Cache;
use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;
use zarrs::array::data_type::{bool, float32, string, uint32};
use zarrs::array::{Array, ArrayBuilder, ArraySubset, FillValueMetadata};
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use crate::build::assign::determine_node_assignments;
use crate::build::builder::TRACKED_MEMORY_FRACTION;
use crate::build::source::EmbeddingsSource;
use crate::build::writer::{build_embeddings_array, store_embeddings_subset, zarrs_append};
use crate::search::Node;
use crate::utils::{EmbeddingDtype, Metric};

/// Writes `info/levels`, `info/metric`, and `info/is_normalized`.
pub fn write_index_info(
    store: &ReadableWritableListableStorage,
    levels: u32,
    metric: Metric,
    is_normalized: bool,
) {
    // Zarr has no bare-scalar type; each of these is a rank-0 array.
    let scalar_shape: Vec<u64> = vec![];

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), uint32(), 0u32)
        .build(store.clone(), "/info/levels")
        .expect("Failed to build info/levels array");
    field
        .store_metadata()
        .expect("Failed to store info/levels metadata");
    field
        .store_chunk(&[], vec![levels])
        .expect("Failed to store info/levels chunk");

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), string(), "")
        .build(store.clone(), "/info/metric")
        .expect("Failed to build info/metric array");
    field
        .store_metadata()
        .expect("Failed to store info/metric metadata");
    field
        .store_chunk(&[], vec![metric.as_str().to_string()])
        .expect("Failed to store info/metric chunk");

    let field = ArrayBuilder::new(
        scalar_shape.clone(),
        scalar_shape,
        bool(),
        FillValueMetadata::Bool(false),
    )
    .build(store.clone(), "/info/is_normalized")
    .expect("Failed to build info/is_normalized array");
    field
        .store_metadata()
        .expect("Failed to store info/is_normalized metadata");
    field
        .store_chunk(&[], vec![is_normalized])
        .expect("Failed to store info/is_normalized chunk");
}

/// Writes `info/{name}` as a rank-0 (scalar) `uint32` array, overwriting it
/// if it already exists. Split out from `write_index_info` since these
/// fields change after construction: `total_items` and `next_item_id` are
/// only known once `build`'s `dataset` is available, and `insert` rewrites
/// both.
pub fn write_info_u32(store: &ReadableWritableListableStorage, name: &str, value: u32) {
    let scalar_shape: Vec<u64> = vec![];
    let path = format!("/info/{name}");

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape, uint32(), 0u32)
        .build(store.clone(), &path)
        .unwrap_or_else(|e| panic!("Failed to build {path} array: {e}"));
    field
        .store_metadata()
        .unwrap_or_else(|e| panic!("Failed to store {path} metadata: {e}"));
    field
        .store_chunk(&[], vec![value])
        .unwrap_or_else(|e| panic!("Failed to store {path} chunk: {e}"));
}

/// Writes `index_root/embeddings`, the top-level cluster leaders. Small by
/// construction, written once, no appending needed.
pub fn write_index_root(
    store: &ReadableWritableListableStorage,
    root_embeddings: &Array2<f32>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) {
    let shape = vec![
        root_embeddings.nrows() as u64,
        root_embeddings.ncols() as u64,
    ];
    let subset = ArraySubset::new_with_ranges(&[0..shape[0], 0..shape[1]]);
    let path = "/index_root/embeddings";
    let array = build_embeddings_array(store, path, shape, chunk_shape, dtype);
    store_embeddings_subset(&array, &subset, root_embeddings, dtype, path);
}

/// Appends a batch to `group_path` (a `lvl_N/node_M` group).
/// Creates its `embeddings`/`child_key`/`border` arrays on the first call
/// for that path, appends to them on every later call.
///
/// `border` is left at its fill value here, never populated.
pub fn append_node_batch(
    store: &ReadableWritableListableStorage,
    group_path: &str,
    child_key: &str,
    embeddings: &Array2<f32>,
    children: &Array1<u32>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) {
    let embeddings_path = format!("{group_path}/embeddings");
    let children_path = format!("{group_path}/{child_key}");
    let is_new = Array::open(store.clone(), &embeddings_path).is_err();

    zarrs_append(
        store,
        &embeddings_path,
        &children_path,
        embeddings,
        children,
        chunk_shape,
        dtype,
    );

    if is_new {
        let border_shape = vec![2u64];
        let border_array = ArrayBuilder::new(border_shape.clone(), border_shape, float32(), 0.0f32)
            .build(store.clone(), &format!("{group_path}/border"))
            .expect("Failed to build border array");
        border_array
            .store_metadata()
            .expect("Failed to store border metadata");
    }
}

/// A cached node's `(centroids, children)`.
type CachedNode = Arc<(Array2<f32>, Array1<u32>)>;

/// Caches a node's `(centroids, children)` across every batch and pass of
/// one `build_tree` (or `Index::insert`) call, so a shallow node isn't
/// re-read from disk every time a deeper pass routes through it. Capacity
/// is fixed at construction; moka evicts by weight (bytes) once it's full.
pub(crate) struct NodeCache {
    // Key: node group_path (e.g. "/lvl_1/node_3"). Value: (centroids, children).
    cache: Cache<String, CachedNode>,
}

impl NodeCache {
    fn entry_bytes(entry: &(Array2<f32>, Array1<u32>)) -> u32 {
        (entry.0.len() * size_of::<f32>() + entry.1.len() * size_of::<u32>()) as u32
    }

    pub(crate) fn new(max_capacity_bytes: u64) -> Self {
        NodeCache {
            cache: Cache::builder()
                .max_capacity(max_capacity_bytes)
                .weigher(|_, v: &CachedNode| Self::entry_bytes(v))
                .build(),
        }
    }

    /// `group_path`'s `(centroids, children)`, from cache or disk.
    fn get_or_read(&self, store: &ReadableWritableListableStorage, group_path: &str) -> CachedNode {
        self.cache.get_with(group_path.to_string(), || {
            let read_store: ReadableListableStorage = store.clone().readable_listable();
            let node = Node::new(read_store, group_path.to_string(), "node_ids".to_string());
            let centroids = node
                .embeddings()
                .as_ref()
                .expect(
                    "node embeddings missing: never written by build, or insert routed into a \
                     branch with zero descendants from the original build",
                )
                .clone();
            let child_ids = node
                .children()
                .as_ref()
                .expect(
                    "node children missing: never written by build, or insert routed into a \
                     branch with zero descendants from the original build",
                )
                .clone();
            Arc::new((centroids, child_ids))
        })
    }
}

/// Parameters that stay constant across one recursive descent, bundled to
/// keep `add_data`/`route_batch_to_node`'s signatures manageable.
pub(crate) struct BuildConfig<'a> {
    pub(crate) store: &'a ReadableWritableListableStorage,
    pub(crate) node_cache: &'a NodeCache,
    pub(crate) target_level: u32,
    pub(crate) total_levels: u32,
    pub(crate) metric: Metric,
    pub(crate) is_normalized: bool,
    pub(crate) chunk_shape: &'a [u64],
    pub(crate) embedding_dtype: EmbeddingDtype,

    /// Locks a leaf during a write. Only set when inserting into an
    /// already-built index; build_tree passes None.
    pub(crate) leaf_locks: Option<&'a DashMap<u32, Arc<RwLock<()>>>>,
}

/// Calls `process(i)` for each `i` in `0..count` and collects the results.
/// Runs on rayon's thread pool, unless `on_caller_thread` is true, in
/// which case every call runs on the thread that called `fan_out`
/// instead.
///
/// Set `on_caller_thread` whenever `process` might wait on a leaf's
/// write lock, to avoid deadlocking rayon's thread pool.
fn fan_out<F>(count: usize, on_caller_thread: bool, process: F) -> Vec<(u32, u32)>
where
    F: Fn(usize) -> Vec<(u32, u32)> + Sync + Send,
{
    if on_caller_thread {
        (0..count).flat_map(process).collect()
    } else {
        (0..count).into_par_iter().flat_map(process).collect()
    }
}

/// Routes a batch of data points (already known to belong under `node_idx`
/// at `level`) toward `config.target_level`: writes them if this is that
/// level, otherwise reads `node_idx`'s own centroids/children (written by
/// an earlier `target_level` pass), splits the batch by nearest centroid,
/// and recurses into each non-empty child.
///
/// Returns the on-disk `(level, node_idx)` of every leaf actually written
/// to, so a caller mutating an already-loaded `Index` can invalidate
/// exactly those cached nodes.
fn route_batch_to_node(
    config: &BuildConfig,
    level: u32,
    node_idx: u32,
    data_embeddings: &Array2<f32>,
    data_ids: &Array1<u32>,
) -> Vec<(u32, u32)> {
    let group_path = format!("/lvl_{level}/node_{node_idx}");

    if level == config.target_level {
        let child_key = if level == config.total_levels {
            "item_ids"
        } else {
            "node_ids"
        };
        let write = || {
            append_node_batch(
                config.store,
                &group_path,
                child_key,
                data_embeddings,
                data_ids,
                config.chunk_shape,
                config.embedding_dtype,
            )
        };
        match config.leaf_locks {
            Some(locks) => {
                let lock = locks
                    .entry(node_idx)
                    .or_insert_with(|| Arc::new(RwLock::new(())))
                    .clone();
                let _guard = lock.write().unwrap();
                write();
            }
            None => write(),
        }
        return vec![(level, node_idx)];
    }

    // Cached across every batch/pass of this build_tree (or insert) call,
    // not re-read from disk on every visit.
    let entry = config.node_cache.get_or_read(config.store, &group_path);
    let (centroids, child_ids) = (&entry.0, &entry.1);

    let (offsets, assignment) = determine_node_assignments(
        centroids,
        data_embeddings,
        config.metric,
        config.is_normalized,
    );

    fan_out(centroids.nrows(), config.leaf_locks.is_some(), |child| {
        let start = offsets[child] as usize;
        let end = offsets[child + 1] as usize;
        if start == end {
            return Vec::new();
        }
        let vec_indices: Vec<usize> = assignment
            .slice(s![start..end])
            .iter()
            .map(|&i| i as usize)
            .collect();
        let child_embeddings = data_embeddings.select(Axis(0), &vec_indices);
        let child_ids_batch = Array1::from_iter(vec_indices.iter().map(|&i| data_ids[i]));
        route_batch_to_node(
            config,
            level + 1,
            child_ids[child],
            &child_embeddings,
            &child_ids_batch,
        )
    })
}

/// Routes a batch of new data toward `config.target_level`: assigns each
/// point to its nearest root centroid, then routes/writes each non-empty
/// group via `route_batch_to_node` starting at level 1. Shared by
/// `build_tree` (one call per pass, `target_level` = that pass's level)
/// and `Index::insert` (one call, `target_level` = `total_levels`).
///
/// Returns the on-disk `(level, node_idx)` of every leaf actually written
/// to.
pub(crate) fn add_data(
    config: &BuildConfig,
    root_embeddings: &Array2<f32>,
    data_embeddings: &Array2<f32>,
    data_ids: &Array1<u32>,
) -> Vec<(u32, u32)> {
    let (offsets, assignment) = determine_node_assignments(
        root_embeddings,
        data_embeddings,
        config.metric,
        config.is_normalized,
    );

    fan_out(
        root_embeddings.nrows(),
        config.leaf_locks.is_some(),
        |root_node| {
            let start = offsets[root_node] as usize;
            let end = offsets[root_node + 1] as usize;
            if start == end {
                return Vec::new();
            }
            let vec_indices: Vec<usize> = assignment
                .slice(s![start..end])
                .iter()
                .map(|&i| i as usize)
                .collect();
            let node_embeddings = data_embeddings.select(Axis(0), &vec_indices);
            let node_ids = Array1::from_iter(vec_indices.iter().map(|&i| data_ids[i]));
            route_batch_to_node(config, 1, root_node as u32, &node_embeddings, &node_ids)
        },
    )
}

/// `build_tree`'s parameters, bundled to keep its own signature manageable.
#[derive(Clone, Copy)]
pub struct BuildTreeArgs<'a> {
    pub store: &'a ReadableWritableListableStorage,
    pub root_embeddings: &'a Array2<f32>,
    pub representatives: &'a EmbeddingsSource,
    pub dataset: &'a EmbeddingsSource,
    pub total_levels: u32,
    pub metric: Metric,
    pub is_normalized: bool,
    pub fallback_batch_vecs: usize,
    pub chunk_shape: &'a [u64],
    pub embedding_dtype: EmbeddingDtype,
    pub memory_limit_bytes: usize,
}

/// Builds every level of the tree under `args.root_embeddings`, one
/// on-disk pass per level. Each non-leaf pass reads only as many
/// representatives as that level needs to end up with `ns` children per
/// node; the last pass reads the full dataset, streamed in batches like
/// every other pass.
///
/// Example, `root_embeddings.nrows() = ns = 100`, `total_levels = 3`,
/// `representatives.shape().0 = R = 1_000_000`:
///
///   target_level=1: reads first ns^2 = 10_000 of `representatives`
///   target_level=2: reads first ns^3 = 1_000_000 of `representatives` (all of R)
///   target_level=3 (== total_levels): reads all of `dataset`
pub fn build_tree(args: &BuildTreeArgs) {
    let BuildTreeArgs {
        store,
        root_embeddings,
        representatives,
        dataset,
        total_levels,
        metric,
        is_normalized,
        fallback_batch_vecs,
        chunk_shape,
        embedding_dtype,
        memory_limit_bytes,
    } = *args;

    let node_size = root_embeddings.nrows() as u64;
    let tracked_budget = (memory_limit_bytes as f64 * TRACKED_MEMORY_FRACTION) as usize;
    let bytes_per_vec = (root_embeddings.ncols() * size_of::<f32>()).max(1);

    // node_cache gets up to 75% of the tracked budget, sized once for the
    // whole build (not ratcheted per pass): the total bytes every non-leaf
    // level will need once fully on disk.
    let total_cacheable_bytes: usize = (1..total_levels)
        .map(|l| node_size.pow(l + 1) as usize * bytes_per_vec)
        .sum();
    let cache_capacity = total_cacheable_bytes.min(tracked_budget * 3 / 4);
    let batch_share = tracked_budget.saturating_sub(cache_capacity);
    let memory_floor_vecs = (batch_share / bytes_per_vec).max(1);
    let node_cache = NodeCache::new(cache_capacity as u64);

    for target_level in 1..=total_levels {
        let source = if target_level == total_levels {
            dataset
        } else {
            representatives
        };
        let (total_vec_count, _dim) = source.shape();
        let vec_count = if target_level == total_levels {
            total_vec_count
        } else {
            (node_size.pow(target_level + 1) as usize).min(total_vec_count)
        };
        let batch_vecs = source.chunk_aligned_batch_vecs(memory_floor_vecs, fallback_batch_vecs);

        let config = BuildConfig {
            store,
            node_cache: &node_cache,
            target_level,
            total_levels,
            metric,
            is_normalized,
            chunk_shape,
            embedding_dtype,
            leaf_locks: None,
        };

        let mut start = 0;
        while start < vec_count {
            let end = (start + batch_vecs).min(vec_count);
            log::debug!("target_level={target_level}: processing vecs {start}..{end}");
            let batch_embeddings = source.read_vecs(start, end);
            let batch_ids: Array1<u32> = (start as u32..end as u32).collect();
            add_data(&config, root_embeddings, &batch_embeddings, &batch_ids);
            start = end;
        }
    }
}

#[cfg(test)]
#[path = "utests/tree.rs"]
mod tests;
