use std::sync::{Arc, Mutex};

use half::f16;
use lru::LruCache;
use ndarray::{s, Array1, Array2, Axis};
use rayon::prelude::*;
use zarrs::array::data_type::{bool, float16, float32, string, uint32};
use zarrs::array::{Array, ArrayBuilder, ArraySubset, FillValueMetadata};
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use crate::build::assign::determine_node_assignments;
use crate::build::builder::TRACKED_MEMORY_FRACTION;
use crate::build::source::EmbeddingsSource;
use crate::build::writer::zarrs_append;
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
    field.store_metadata().expect("Failed to store info/levels metadata");
    field.store_chunk(&[], vec![levels]).expect("Failed to store info/levels chunk");

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), string(), "")
        .build(store.clone(), "/info/metric")
        .expect("Failed to build info/metric array");
    field.store_metadata().expect("Failed to store info/metric metadata");
    field
        .store_chunk(&[], vec![metric.as_str().to_string()])
        .expect("Failed to store info/metric chunk");

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape, bool(), FillValueMetadata::Bool(false))
        .build(store.clone(), "/info/is_normalized")
        .expect("Failed to build info/is_normalized array");
    field.store_metadata().expect("Failed to store info/is_normalized metadata");
    field
        .store_chunk(&[], vec![is_normalized])
        .expect("Failed to store info/is_normalized chunk");
}

/// Writes `index_root/embeddings`, the top-level cluster leaders. Small by
/// construction, written once, no appending needed.
pub fn write_index_root(
    store: &ReadableWritableListableStorage,
    root_embeddings: &Array2<f32>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) {
    let shape = vec![root_embeddings.nrows() as u64, root_embeddings.ncols() as u64];
    let subset = ArraySubset::new_with_ranges(&[0..shape[0], 0..shape[1]]);
    match dtype {
        EmbeddingDtype::F32 => {
            let mut builder = ArrayBuilder::new(shape.clone(), chunk_shape.to_vec(), float32(), 0.0f32);
            builder.bytes_to_bytes_codecs(crate::build::writer::compressor());
            let array = builder
                .build(store.clone(), "/index_root/embeddings")
                .expect("Failed to build index_root/embeddings array");
            array.store_metadata().expect("Failed to store index_root/embeddings metadata");
            array.store_array_subset(&subset, root_embeddings).expect("Failed to store index_root/embeddings");
        }
        EmbeddingDtype::F16 => {
            let mut builder = ArrayBuilder::new(shape.clone(), chunk_shape.to_vec(), float16(), f16::from_f32(0.0));
            builder.bytes_to_bytes_codecs(crate::build::writer::compressor());
            let array = builder
                .build(store.clone(), "/index_root/embeddings")
                .expect("Failed to build index_root/embeddings array");
            array.store_metadata().expect("Failed to store index_root/embeddings metadata");
            array
                .store_array_subset(&subset, &root_embeddings.mapv(f16::from_f32))
                .expect("Failed to store index_root/embeddings");
        }
    }
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

    zarrs_append(store, &embeddings_path, &children_path, embeddings, children, chunk_shape, dtype);

    if is_new {
        let border_shape = vec![2u64];
        let border_array = ArrayBuilder::new(border_shape.clone(), border_shape, float32(), 0.0f32)
            .build(store.clone(), &format!("{group_path}/border"))
            .expect("Failed to build border array");
        border_array.store_metadata().expect("Failed to store border metadata");
    }
}

/// Caches a node's `(centroids, children)` across every batch and pass of
/// one `build_tree` call, so a shallow node isn't re-read from disk every
/// time a deeper pass routes through it. Unbounded until `set_limit` gives
/// it a real budget (adjusted per pass, since it depends on that pass's
/// batch size); least-recently-used entries are evicted first.
struct NodeCache {
    state: Mutex<(LruCache<String, Arc<(Array2<f32>, Array1<u32>)>>, usize, usize)>,
}

impl NodeCache {
    fn new() -> Self {
        NodeCache { state: Mutex::new((LruCache::unbounded(), 0, usize::MAX)) }
    }

    fn entry_bytes(entry: &(Array2<f32>, Array1<u32>)) -> usize {
        entry.0.len() * size_of::<f32>() + entry.1.len() * size_of::<u32>()
    }

    /// Evicts down to `limit_bytes` if needed, then adopts it as the limit
    /// for future inserts.
    fn set_limit(&self, limit_bytes: usize) {
        let mut state = self.state.lock().unwrap();
        state.2 = limit_bytes;
        while state.1 > state.2 {
            let Some((_, evicted)) = state.0.pop_lru() else { break };
            state.1 -= Self::entry_bytes(&evicted);
        }
    }

    /// `group_path`'s `(centroids, children)`, from cache or disk.
    fn get_or_read(&self, store: &ReadableWritableListableStorage, group_path: &str) -> Arc<(Array2<f32>, Array1<u32>)> {
        if let Some(hit) = self.state.lock().unwrap().0.get(group_path) {
            return hit.clone();
        }

        let read_store: ReadableListableStorage = store.clone().readable_listable();
        let mut node = Node::new(read_store, group_path.to_string(), "node_ids".to_string());
        let centroids = node
            .embeddings()
            .as_ref()
            .expect("intermediate node must already have embeddings from an earlier level pass")
            .clone();
        let child_ids = node
            .children()
            .as_ref()
            .expect("intermediate node must already have children from an earlier level pass")
            .clone();
        let entry = Arc::new((centroids, child_ids));

        let mut state = self.state.lock().unwrap();
        if let Some(hit) = state.0.get(group_path) {
            return hit.clone();
        }
        state.1 += Self::entry_bytes(&entry);
        state.0.put(group_path.to_string(), entry.clone());
        while state.1 > state.2 {
            let Some((_, evicted)) = state.0.pop_lru() else { break };
            state.1 -= Self::entry_bytes(&evicted);
        }
        entry
    }
}

/// Parameters that stay constant across one `build_tree` call's whole
/// recursive descent, bundled to keep `add_data`'s signature manageable.
struct BuildConfig<'a> {
    store: &'a ReadableWritableListableStorage,
    node_cache: &'a NodeCache,
    target_level: u32,
    total_levels: u32,
    metric: Metric,
    is_normalized: bool,
    chunk_shape: &'a [u64],
    embedding_dtype: EmbeddingDtype,
}

/// Routes a batch of data points (already known to belong under `node_idx`
/// at `level`) toward `config.target_level`: writes them if this is that
/// level, otherwise reads `node_idx`'s own centroids/children (written by
/// an earlier `target_level` pass), splits the batch by nearest centroid,
/// and recurses into each non-empty child.
fn add_data(
    config: &BuildConfig,
    level: u32,
    node_idx: u32,
    data_embeddings: &Array2<f32>,
    data_ids: &Array1<u32>
) {
    let group_path = format!("/lvl_{level}/node_{node_idx}");

    if level == config.target_level {
        let child_key = if level == config.total_levels { "item_ids" } else { "node_ids" };
        append_node_batch(
            config.store,
            &group_path,
            child_key,
            data_embeddings,
            data_ids,
            config.chunk_shape,
            config.embedding_dtype
        );
        return;
    }

    // Cached across every batch/pass of this build_tree call, not re-read
    // from disk on every visit.
    let entry = config.node_cache.get_or_read(config.store, &group_path);
    let (centroids, child_ids) = (&entry.0, &entry.1);

    let (offsets, assignment) =
        determine_node_assignments(
            centroids,
            data_embeddings,
            config.metric,
            config.is_normalized
        );

    (0..centroids.nrows()).into_par_iter().for_each(|child| {
        let start = offsets[child] as usize;
        let end = offsets[child + 1] as usize;
        if start == end {
            return;
        }
        let vec_indices: Vec<usize> = assignment.slice(s![start..end]).iter().map(|&i| i as usize).collect();
        let child_embeddings = data_embeddings.select(Axis(0), &vec_indices);
        let child_ids_batch = Array1::from_iter(vec_indices.iter().map(|&i| data_ids[i]));
        add_data(config, level + 1, child_ids[child], &child_embeddings, &child_ids_batch);
    });
}

/// Builds every level of the tree under `root_embeddings`, one on-disk
/// pass per level. Each non-leaf pass reads only as many representatives
/// as that level needs to end up with `ns` children per node; the last
/// pass reads the full dataset, streamed in batches like every other pass.
///
/// Example, `root_embeddings.nrows() = ns = 100`, `total_levels = 3`,
/// `representatives.shape().0 = R = 1_000_000`:
///
///   target_level=1: reads first ns^2 = 10_000 of `representatives`
///   target_level=2: reads first ns^3 = 1_000_000 of `representatives` (all of R)
///   target_level=3 (== total_levels): reads all of `dataset`
pub fn build_tree(
    store: &ReadableWritableListableStorage,
    root_embeddings: &Array2<f32>,
    representatives: &EmbeddingsSource,
    dataset: &EmbeddingsSource,
    total_levels: u32,
    metric: Metric,
    is_normalized: bool,
    fallback_batch_vecs: usize,
    chunk_shape: &[u64],
    embedding_dtype: EmbeddingDtype,
    memory_limit_bytes: usize,
) {
    let node_size = root_embeddings.nrows() as u64;
    let node_cache = NodeCache::new();
    let tracked_budget = (memory_limit_bytes as f64 * TRACKED_MEMORY_FRACTION) as usize;
    let bytes_per_vec = (root_embeddings.ncols() * size_of::<f32>()).max(1);
    let mut nodes_bytes_needed = 0usize;

    for target_level in 1..=total_levels {
        let source = if target_level == total_levels { dataset } else { representatives };
        let (total_vec_count, _dim) = source.shape();
        let vec_count = if target_level == total_levels {
            total_vec_count
        } else {
            (node_size.pow(target_level + 1) as usize).min(total_vec_count)
        };

        // node_cache gets up to 75% budget
        let batch_share = if nodes_bytes_needed > tracked_budget * 3 / 4 {
            tracked_budget / 4
        } else {
            tracked_budget - nodes_bytes_needed
        };
        let memory_floor_vecs = (batch_share / bytes_per_vec).max(1);
        let batch_vecs = source.chunk_aligned_batch_vecs(memory_floor_vecs, fallback_batch_vecs);
        node_cache.set_limit(tracked_budget.saturating_sub(batch_vecs * bytes_per_vec));

        let config =
            BuildConfig {
                store,
                node_cache: &node_cache,
                target_level,
                total_levels,
                metric,
                is_normalized,
                chunk_shape,
                embedding_dtype
            };

        let mut start = 0;
        while start < vec_count {
            let end = (start + batch_vecs).min(vec_count);
            log::debug!("target_level={target_level}: processing vecs {start}..{end}");
            let batch_embeddings = source.read_vecs(start, end);
            let batch_ids: Array1<u32> = (start as u32..end as u32).collect();

            let (offsets, assignment) =
                determine_node_assignments(root_embeddings, &batch_embeddings, metric, is_normalized);

            (0..root_embeddings.nrows()).into_par_iter().for_each(|root_node| {
                let s = offsets[root_node] as usize;
                let e = offsets[root_node + 1] as usize;
                if s == e {
                    return;
                }
                let vec_indices: Vec<usize> = assignment.slice(s![s..e]).iter().map(|&i| i as usize).collect();
                let node_embeddings = batch_embeddings.select(Axis(0), &vec_indices);
                let node_ids = Array1::from_iter(vec_indices.iter().map(|&i| batch_ids[i]));
                add_data(&config, 1, root_node as u32, &node_embeddings, &node_ids);
            });

            start = end;
        }

        // This level is now on disk, so it's cacheable for every later pass.
        if target_level < total_levels {
            nodes_bytes_needed += node_size.pow(target_level + 1) as usize * bytes_per_vec;
        }
    }
}

#[cfg(test)]
#[path = "utests/tree.rs"]
mod tests;
