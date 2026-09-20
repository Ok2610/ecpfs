use std::sync::{Arc, RwLock};

use dashmap::DashMap;
use moka::sync::Cache;
use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use crate::build::assign::determine_node_assignments;
use crate::build::builder::TRACKED_MEMORY_FRACTION;
use crate::build::source::EmbeddingsSource;
use crate::build::writer::append_node_batch;
use crate::dtype::EmbeddingDtype;
use crate::metric::Metric;
use crate::search::Node;

/// A cached node's `(representatives, children)`.
type CachedNode = Arc<(Array2<f32>, Array1<u32>)>;

/// Caches internal nodes' `(representatives, children)` during one build or insert,
/// so routing each batch down the tree doesn't re-read them from disk.
pub(crate) struct NodeCache {
    /// Keyed by group path, such as `/lvl_1/node_3`.
    cache: Cache<String, CachedNode>,
}

impl NodeCache {
    /// Returns the bytes an entry takes, which the cache counts against its capacity.
    fn entry_bytes(entry: &(Array2<f32>, Array1<u32>)) -> u32 {
        (entry.0.len() * size_of::<f32>() + entry.1.len() * size_of::<u32>()) as u32
    }

    /// Creates an empty cache holding at most `max_capacity_bytes`.
    pub(crate) fn new(max_capacity_bytes: u64) -> Self {
        NodeCache {
            cache: Cache::builder()
                .max_capacity(max_capacity_bytes)
                .weigher(|_, v: &CachedNode| Self::entry_bytes(v))
                .build(),
        }
    }

    /// Gets a node's representatives and children, reading from disk only on a cache miss.
    fn get_or_read(&self, store: &ReadableWritableListableStorage, group_path: &str) -> CachedNode {
        self.cache.get_with(group_path.to_string(), || {
            let read_store: ReadableListableStorage = store.clone().readable_listable();
            let node = Node::new(read_store, group_path.to_string(), "node_ids".to_string());
            let representatives = node
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
            Arc::new((representatives, child_ids))
        })
    }
}

/// The settings `add_data` passes unchanged down the tree.
pub(crate) struct BuildConfig<'a> {
    pub(crate) store: &'a ReadableWritableListableStorage,
    pub(crate) node_cache: &'a NodeCache,
    /// The 1-based level written to; levels above it are only routed through.
    pub(crate) target_level: u32,
    pub(crate) total_levels: u32,
    pub(crate) metric: Metric,
    pub(crate) is_normalized: bool,
    pub(crate) chunk_shape: &'a [u64],
    pub(crate) embedding_dtype: EmbeddingDtype,

    /// Per-leaf locks, taken around each write. `None` when nothing else can
    /// read the index yet, as during a build.
    pub(crate) leaf_locks: Option<&'a DashMap<u32, Arc<RwLock<()>>>>,
}

/// Calls `process(i)` for each `i` in `0..count` in parallel and collects the results.
/// Set `on_caller_thread` to run them one at a time on this thread instead. Do so
/// when `process` may wait on a leaf lock, since that can deadlock rayon's pool.
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

/// Routes a batch of vectors that belong under node `node_idx` at `level` down
/// to `config.target_level`, and appends each to its nearest node there.
/// Returns the on-disk `(level, node_idx)` of every node written to.
fn route_batch_to_node(
    config: &BuildConfig,
    level: u32,
    node_idx: u32,
    data_embeddings: &Array2<f32>,
    data_ids: &Array1<u32>,
) -> Vec<(u32, u32)> {
    let group_path = format!("/lvl_{level}/node_{node_idx}");

    // At the target level, append the batch (under the leaf's lock, if any)
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

    // Above the target level, split the batch by nearest child and recurse into each
    let entry = config.node_cache.get_or_read(config.store, &group_path);
    let (representatives, child_ids) = (&entry.0, &entry.1);

    let (offsets, assignment) = determine_node_assignments(
        representatives,
        data_embeddings,
        config.metric,
        config.is_normalized,
    );

    fan_out(
        representatives.nrows(),
        config.leaf_locks.is_some(),
        |child| {
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
        },
    )
}

/// Adds vectors to the tree at `config.target_level`. Each row of `data_embeddings`
/// goes down from the root to its nearest node there, along with its id from
/// `data_ids`. Returns the on-disk `(level, node_idx)` of every node written to.
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
    /// Every representative, starting with the root's. The non-leaf levels are
    /// built from these.
    pub representatives: &'a EmbeddingsSource,
    /// The items the leaves hold.
    pub dataset: &'a EmbeddingsSource,
    pub total_levels: u32,
    pub metric: Metric,
    pub is_normalized: bool,
    /// The chunk size assumed for a source that isn't chunked.
    pub fallback_batch_vecs: usize,
    pub chunk_shape: &'a [u64],
    pub embedding_dtype: EmbeddingDtype,
    pub memory_limit_bytes: usize,
}

/// Builds the tree one level at a time from the top, appending each input
/// vector to its nearest node on that level. Level `l` reads the first
/// `ns^(l+1)` representatives, where `ns` is the number of root entries; the
/// leaf level reads `dataset`.
///
/// Example, `root_embeddings.nrows() = ns = 100`, `total_levels = 3`,
/// `representatives.shape().0 = R = 1_000_000`:
///
/// ```text
/// target_level=1: reads first ns^2 = 10_000 of `representatives`
/// target_level=2: reads first ns^3 = 1_000_000 of `representatives` (all of R)
/// target_level=3 (== total_levels): reads all of `dataset`
/// ```
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

    // The node cache gets what every non-leaf level needs, up to 3/4 of the
    // budget; batches get the rest.
    let total_cacheable_bytes: usize = (1..total_levels)
        .map(|l| node_size.pow(l + 1) as usize * bytes_per_vec)
        .sum();
    let cache_capacity = total_cacheable_bytes.min(tracked_budget * 3 / 4);
    let batch_share = tracked_budget.saturating_sub(cache_capacity);
    let memory_floor_vecs = (batch_share / bytes_per_vec).max(1);
    let node_cache = NodeCache::new(cache_capacity as u64);

    for target_level in 1..=total_levels {
        // Non-leaf levels are built from representatives, the leaf level from the dataset
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

        // Row indices are item ids on the leaf level, and representative ids
        // (the next level's node ids) above it

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
