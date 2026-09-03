use ndarray::{s, Array1, Array2, Axis};
use rayon::prelude::*;
use zarrs::array::data_type::{bool, float32, string, uint32};
use zarrs::array::{Array, ArrayBuilder, ArraySubset, FillValueMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use crate::build::assign::determine_node_assignments;
use crate::build::source::EmbeddingsSource;
use crate::build::writer::zarrs_append;
use crate::search::Node;
use crate::utils::Metric;

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

/// Writes `index_root/embeddings` (the top-level cluster leaders - small by
/// construction, written once, no appending needed).
pub fn write_index_root(store: &ReadableWritableListableStorage, root_embeddings: &Array2<f32>, chunk_shape: &[u64]) {
    let shape = vec![root_embeddings.nrows() as u64, root_embeddings.ncols() as u64];
    let mut builder = ArrayBuilder::new(shape.clone(), chunk_shape.to_vec(), float32(), 0.0f32);
    builder.bytes_to_bytes_codecs(crate::build::writer::compressor());
    let array = builder
        .build(store.clone(), "/index_root/embeddings")
        .expect("Failed to build index_root/embeddings array");
    array.store_metadata().expect("Failed to store index_root/embeddings metadata");
    array
        .store_array_subset(&ArraySubset::new_with_ranges(&[0..shape[0], 0..shape[1]]), root_embeddings)
        .expect("Failed to store index_root/embeddings");
}

/// Appends a batch of rows to `group_path` (a `lvl_N/node_M` group) -
/// creates its `embeddings`/`child_key`/`border` arrays on the first call
/// for that path, appends to them on every later call.
///
/// `border` is reserved for a future pruning feature; it's left at its
/// fill value here, never populated.
pub fn append_node_batch(
    store: &ReadableWritableListableStorage,
    group_path: &str,
    child_key: &str,
    embeddings: &Array2<f32>,
    children: &Array1<u32>,
    chunk_shape: &[u64],
) {
    let embeddings_path = format!("{group_path}/embeddings");
    let children_path = format!("{group_path}/{child_key}");
    let is_new = Array::open(store.clone(), &embeddings_path).is_err();

    zarrs_append(store, &embeddings_path, &children_path, embeddings, children, chunk_shape);

    if is_new {
        let border_shape = vec![2u64];
        let border_array = ArrayBuilder::new(border_shape.clone(), border_shape, float32(), 0.0f32)
            .build(store.clone(), &format!("{group_path}/border"))
            .expect("Failed to build border array");
        border_array.store_metadata().expect("Failed to store border metadata");
    }
}

/// Parameters that stay constant across one `build_tree` call's whole
/// recursive descent, bundled to keep `add_data`'s signature manageable.
struct BuildConfig<'a> {
    store: &'a ReadableWritableListableStorage,
    target_level: u32,
    total_levels: u32,
    metric: Metric,
    is_normalized: bool,
    chunk_shape: &'a [u64],
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
    data_ids: &Array1<u32>) {
    let group_path = format!("/lvl_{level}/node_{node_idx}");

    if level == config.target_level {
        let child_key = if level == config.total_levels { "item_ids" } else { "node_ids" };
        append_node_batch(
            config.store, 
            &group_path,
            child_key,
            data_embeddings,
            data_ids,
            config.chunk_shape
        );
        return;
    }

    let mut node = Node::new(
        config.store.clone().readable_listable(),
        group_path, 
        "node_ids".to_string()
    );
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

    let (offsets, assignment) =
        determine_node_assignments(
            &centroids, 
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
        let rows: Vec<usize> = assignment.slice(s![start..end]).iter().map(|&i| i as usize).collect();
        let child_embeddings = data_embeddings.select(Axis(0), &rows);
        let child_ids_batch = Array1::from_iter(rows.iter().map(|&i| data_ids[i]));
        add_data(config, level + 1, child_ids[child], &child_embeddings, &child_ids_batch);
    });
}

/// Builds every level of the tree under `root_embeddings`: for each level
/// but the last, redistributes `representatives` (the full representative
/// set); for the last (leaf) level, redistributes `dataset` the same way -
/// both one on-disk chunk at a time. A level's nodes must already exist
/// (written by the previous level's pass) before the next level can read
/// their centroids to descend further, so levels build strictly in order.
pub fn build_tree(
    store: &ReadableWritableListableStorage,
    root_embeddings: &Array2<f32>,
    representatives: &EmbeddingsSource,
    dataset: &EmbeddingsSource,
    total_levels: u32,
    metric: Metric,
    is_normalized: bool,
    fallback_batch_rows: usize,
    chunk_shape: &[u64],
) {
    for target_level in 1..=total_levels {
        let source = if target_level == total_levels { dataset } else { representatives };
        let (row_count, _dim) = source.shape();
        let batch_rows = source.natural_batch_rows(fallback_batch_rows);
        let config = 
            BuildConfig { 
                store,
                target_level,
                total_levels, metric,
                is_normalized,
                chunk_shape 
            };

        let mut start = 0;
        while start < row_count {
            let end = (start + batch_rows).min(row_count);
            let batch_embeddings = source.read_rows(start, end);
            let batch_ids: Array1<u32> = (start as u32..end as u32).collect();

            let (offsets, assignment) =
                determine_node_assignments(root_embeddings, &batch_embeddings, metric, is_normalized);

            (0..root_embeddings.nrows()).into_par_iter().for_each(|root_node| {
                let s = offsets[root_node] as usize;
                let e = offsets[root_node + 1] as usize;
                if s == e {
                    return;
                }
                let rows: Vec<usize> = assignment.slice(s![s..e]).iter().map(|&i| i as usize).collect();
                let node_embeddings = batch_embeddings.select(Axis(0), &rows);
                let node_ids = Array1::from_iter(rows.iter().map(|&i| batch_ids[i]));
                add_data(&config, 1, root_node as u32, &node_embeddings, &node_ids);
            });

            start = end;
        }
    }
}

#[cfg(test)]
#[path = "utests/tree.rs"]
mod tests;
