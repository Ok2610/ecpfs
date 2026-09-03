use ndarray::{Array1, Array2};
use zarrs::array::data_type::{bool, float32, string, uint32};
use zarrs::array::{Array, ArrayBuilder, ArraySubset, FillValueMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use crate::build::writer::zarrs_append;
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

#[cfg(test)]
#[path = "utests/tree.rs"]
mod tests;
