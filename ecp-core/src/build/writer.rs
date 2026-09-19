use std::sync::Arc;

use half::f16;
use ndarray::{Array1, Array2};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::data_type::{bool, float16, float32, int8, string, uint8, uint32};
use zarrs::array::{
    Array, ArrayBuilder, ArraySubset, BytesToBytesCodecTraits, DataType, FillValue,
    FillValueMetadata,
};
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use crate::dtype::EmbeddingDtype;
use crate::metric::Metric;

/// Without a compressor, a chunk is padded to its full declared size on
/// disk regardless of how much of it is actually written. Since a chunk is
/// sized by I/O throughput rather than expected data size (the whole point,
/// given eCP doesn't enforce cluster sizes), that padding can be the
/// difference between a few KB and tens of MB per mostly-empty node. zstd
/// compresses the fill-value padding away.
pub(super) fn compressor() -> Vec<Arc<dyn BytesToBytesCodecTraits>> {
    vec![Arc::new(ZstdCodec::new(3, false))]
}

/// Builds `path`'s embeddings array with `dtype`'s zarr type and a zero fill
/// value, zstd-compressed, and stores its metadata.
pub(crate) fn build_embeddings_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: Vec<u64>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) -> Array<dyn ReadableWritableListableStorageTraits> {
    let (data_type, fill_value): (DataType, FillValue) = match dtype {
        EmbeddingDtype::F32 => (float32(), 0.0f32.into()),
        EmbeddingDtype::F16 => (float16(), f16::from_f32(0.0).into()),
        EmbeddingDtype::UInt8 => (uint8(), 0u8.into()),
        EmbeddingDtype::Int8 => (int8(), 0i8.into()),
    };
    let mut builder = ArrayBuilder::new(shape, chunk_shape.to_vec(), data_type, fill_value);
    builder.bytes_to_bytes_codecs(compressor());
    let array = builder
        .build(store.clone(), path)
        .unwrap_or_else(|e| panic!("Failed to build {path} array: {e}"));
    array
        .store_metadata()
        .unwrap_or_else(|e| panic!("Failed to store {path} metadata: {e}"));
    array
}

/// Stores `embeddings` into `subset`, narrowing f32 down to `dtype` as
/// stored. Float-to-int casts saturate rather than wrap, so a value outside
/// the target's range clamps to its nearest end.
pub(crate) fn store_embeddings_subset(
    array: &Array<dyn ReadableWritableListableStorageTraits>,
    subset: &ArraySubset,
    embeddings: &Array2<f32>,
    dtype: EmbeddingDtype,
    context: &str,
) {
    let fail = |e| panic!("Failed to store {context}: {e}");
    match dtype {
        EmbeddingDtype::F32 => array
            .store_array_subset(subset, embeddings)
            .unwrap_or_else(fail),
        EmbeddingDtype::F16 => array
            .store_array_subset(subset, embeddings.mapv(f16::from_f32))
            .unwrap_or_else(fail),
        EmbeddingDtype::UInt8 => array
            .store_array_subset(subset, embeddings.mapv(|x| x as u8))
            .unwrap_or_else(fail),
        EmbeddingDtype::Int8 => array
            .store_array_subset(subset, embeddings.mapv(|x| x as i8))
            .unwrap_or_else(fail),
    }
}

/// Creates (on the first call for a given path) or grows and appends to a
/// paired embeddings+ids array. `dtype` sets the embeddings array's stored
/// width; the ids array is always uint32. `chunk_shape` only applies on
/// creation; a later call's value is ignored once the array exists. Used for
/// a node's `embeddings`/`child_key`, or the representative set's
/// `rep_embeddings`/`rep_item_ids`.
// clippy's single_range_in_vec_init fix would replace the ids array's
// single-element range array with `.collect::<Vec<u64>>()`. That allocates
// on the heap on every call. The array literal here does not, and matches
// the multi-dimensional embeddings ranges elsewhere in this function.
#[allow(clippy::single_range_in_vec_init)]
pub fn zarrs_append(
    store: &ReadableWritableListableStorage,
    embeddings_path: &str,
    ids_path: &str,
    embeddings: &Array2<f32>,
    ids: &Array1<u32>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) {
    match Array::open(store.clone(), embeddings_path) {
        Ok(mut array) => {
            let existing_vecs = array.shape()[0];
            let dim = array.shape()[1];
            let new_vecs = existing_vecs + embeddings.nrows() as u64;
            array
                .set_shape(vec![new_vecs, dim])
                .expect("Failed to grow embeddings array");
            array
                .store_metadata()
                .expect("Failed to store embeddings metadata");
            let subset = ArraySubset::new_with_ranges(&[existing_vecs..new_vecs, 0..dim]);
            store_embeddings_subset(&array, &subset, embeddings, dtype, embeddings_path);

            let mut ids_array = Array::open(store.clone(), ids_path)
                .expect("ids array missing alongside embeddings");
            let existing_ids = ids_array.shape()[0];
            let new_ids = existing_ids + ids.len() as u64;
            ids_array
                .set_shape(vec![new_ids])
                .expect("Failed to grow ids array");
            ids_array
                .store_metadata()
                .expect("Failed to store ids metadata");
            ids_array
                .store_array_subset(&ArraySubset::new_with_ranges(&[existing_ids..new_ids]), ids)
                .expect("Failed to append ids");
        }
        Err(_) => {
            let dim = embeddings.ncols() as u64;
            let emb_shape = vec![embeddings.nrows() as u64, dim];
            let subset = ArraySubset::new_with_ranges(&[0..emb_shape[0], 0..dim]);
            let emb_array =
                build_embeddings_array(store, embeddings_path, emb_shape, chunk_shape, dtype);
            store_embeddings_subset(&emb_array, &subset, embeddings, dtype, embeddings_path);

            // Same chunk-count as the embeddings array, so a given
            // chunk index lines up across both.
            let ids_shape = vec![ids.len() as u64];
            let mut ids_builder =
                ArrayBuilder::new(ids_shape.clone(), vec![chunk_shape[0]], uint32(), 0u32);
            ids_builder.bytes_to_bytes_codecs(compressor());
            let ids_array = ids_builder
                .build(store.clone(), ids_path)
                .expect("Failed to build ids array");
            ids_array
                .store_metadata()
                .expect("Failed to store ids metadata");
            ids_array
                .store_array_subset(&ArraySubset::new_with_ranges(&[0..ids_shape[0]]), ids)
                .expect("Failed to store ids");
        }
    }
}

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

#[cfg(test)]
#[path = "utests/writer.rs"]
mod tests;
