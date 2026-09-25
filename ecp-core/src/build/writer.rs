use std::sync::Arc;

use half::f16;
use ndarray::{Array1, Array2};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::data_type::{bool, float16, float32, int8, string, uint8, uint32};
use zarrs::array::{
    Array, ArrayBuilder, ArrayCreateError, ArraySubset, BytesToBytesCodecTraits, DataType,
    FillValue, FillValueMetadata,
};
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use crate::dtype::EmbeddingDtype;
use crate::error::{EcpError, Result, ResultExt};
use crate::format::FORMAT_VERSION;
use crate::metric::Metric;

/// Compresses every written array with zstd at level 3. Chunks are sized for
/// I/O rather than a node's data, so most are largely empty padding, which
/// zstd shrinks to almost nothing on disk.
pub(super) fn compressor() -> Vec<Arc<dyn BytesToBytesCodecTraits>> {
    vec![Arc::new(ZstdCodec::new(3, false))]
}

/// Creates the embeddings array at `path` with `shape`, stored as `dtype`,
/// zero-filled and zstd-compressed, and returns it for writing.
pub(crate) fn build_embeddings_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: Vec<u64>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) -> Result<Array<dyn ReadableWritableListableStorageTraits>> {
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
        .store_err_with(|| format!("failed to build {path} array"))?;
    array
        .store_metadata()
        .store_err_with(|| format!("failed to store {path} metadata"))?;
    Ok(array)
}

/// Writes `embeddings` into the `subset` region of `array`, converted to
/// `dtype`. Integer dtypes truncate fractions and clamp out-of-range values.
/// `context` names the array in error messages.
pub(crate) fn store_embeddings_subset(
    array: &Array<dyn ReadableWritableListableStorageTraits>,
    subset: &ArraySubset,
    embeddings: &Array2<f32>,
    dtype: EmbeddingDtype,
    context: &str,
) -> Result<()> {
    let context = format!("failed to store {context}");
    match dtype {
        EmbeddingDtype::F32 => array
            .store_array_subset(subset, embeddings)
            .store_err(&context),
        EmbeddingDtype::F16 => array
            .store_array_subset(subset, embeddings.mapv(f16::from_f32))
            .store_err(&context),
        EmbeddingDtype::UInt8 => array
            .store_array_subset(subset, embeddings.mapv(|x| x as u8))
            .store_err(&context),
        EmbeddingDtype::Int8 => array
            .store_array_subset(subset, embeddings.mapv(|x| x as i8))
            .store_err(&context),
    }
}

/// Appends `embeddings` (stored as `dtype`) to the array at `embeddings_path`
/// and `ids` to the one at `ids_path`, creating both on the first call.
/// `chunk_shape` only applies on creation.
// The ids arrays' `&[a..b]` is intended, one range for their single dimension.
#[allow(clippy::single_range_in_vec_init)]
pub fn zarrs_append(
    store: &ReadableWritableListableStorage,
    embeddings_path: &str,
    ids_path: &str,
    embeddings: &Array2<f32>,
    ids: &Array1<u32>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) -> Result<()> {
    match Array::open(store.clone(), embeddings_path) {
        // Grow both arrays, then write into the new rows
        Ok(mut array) => {
            let existing_vecs = array.shape()[0];
            let dim = array.shape()[1];
            let new_vecs = existing_vecs + embeddings.nrows() as u64;
            array
                .set_shape(vec![new_vecs, dim])
                .store_err("failed to grow embeddings array")?;
            array
                .store_metadata()
                .store_err("failed to store embeddings metadata")?;
            let subset = ArraySubset::new_with_ranges(&[existing_vecs..new_vecs, 0..dim]);
            store_embeddings_subset(&array, &subset, embeddings, dtype, embeddings_path)?;

            let mut ids_array = Array::open(store.clone(), ids_path)
                .store_err("ids array missing alongside embeddings")?;
            let existing_ids = ids_array.shape()[0];
            let new_ids = existing_ids + ids.len() as u64;
            ids_array
                .set_shape(vec![new_ids])
                .store_err("failed to grow ids array")?;
            ids_array
                .store_metadata()
                .store_err("failed to store ids metadata")?;
            ids_array
                .store_array_subset(&ArraySubset::new_with_ranges(&[existing_ids..new_ids]), ids)
                .store_err("failed to append ids")?;
        }
        // On the first call, create both arrays
        Err(ArrayCreateError::MissingMetadata) => {
            let dim = embeddings.ncols() as u64;
            let emb_shape = vec![embeddings.nrows() as u64, dim];
            let subset = ArraySubset::new_with_ranges(&[0..emb_shape[0], 0..dim]);
            let emb_array =
                build_embeddings_array(store, embeddings_path, emb_shape, chunk_shape, dtype)?;
            store_embeddings_subset(&emb_array, &subset, embeddings, dtype, embeddings_path)?;

            // Same chunk-count as the embeddings array, so a given
            // chunk index lines up across both.
            let ids_shape = vec![ids.len() as u64];
            let mut ids_builder =
                ArrayBuilder::new(ids_shape.clone(), vec![chunk_shape[0]], uint32(), 0u32);
            ids_builder.bytes_to_bytes_codecs(compressor());
            let ids_array = ids_builder
                .build(store.clone(), ids_path)
                .store_err("failed to build ids array")?;
            ids_array
                .store_metadata()
                .store_err("failed to store ids metadata")?;
            ids_array
                .store_array_subset(&ArraySubset::new_with_ranges(&[0..ids_shape[0]]), ids)
                .store_err("failed to store ids")?;
        }
        Err(e) => {
            return Err(EcpError::Store(format!(
                "failed to open {embeddings_path}: {e}"
            )));
        }
    }
    Ok(())
}

/// Writes `info/format_version`, `info/levels`, `info/metric`, and `info/is_normalized`.
pub fn write_index_info(
    store: &ReadableWritableListableStorage,
    levels: u32,
    metric: Metric,
    is_normalized: bool,
) -> Result<()> {
    write_info_u32(store, "format_version", FORMAT_VERSION)?;

    // Zarr has no bare-scalar type; each of these is a rank-0 array.
    let scalar_shape: Vec<u64> = vec![];

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), uint32(), 0u32)
        .build(store.clone(), "/info/levels")
        .store_err("failed to build info/levels array")?;
    field
        .store_metadata()
        .store_err("failed to store info/levels metadata")?;
    field
        .store_chunk(&[], vec![levels])
        .store_err("failed to store info/levels chunk")?;

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape.clone(), string(), "")
        .build(store.clone(), "/info/metric")
        .store_err("failed to build info/metric array")?;
    field
        .store_metadata()
        .store_err("failed to store info/metric metadata")?;
    field
        .store_chunk(&[], vec![metric.as_str().to_string()])
        .store_err("failed to store info/metric chunk")?;

    let field = ArrayBuilder::new(
        scalar_shape.clone(),
        scalar_shape,
        bool(),
        FillValueMetadata::Bool(false),
    )
    .build(store.clone(), "/info/is_normalized")
    .store_err("failed to build info/is_normalized array")?;
    field
        .store_metadata()
        .store_err("failed to store info/is_normalized metadata")?;
    field
        .store_chunk(&[], vec![is_normalized])
        .store_err("failed to store info/is_normalized chunk")?;
    Ok(())
}

/// Writes `info/{name}` as a `u32` scalar, replacing any existing value.
pub fn write_info_u32(
    store: &ReadableWritableListableStorage,
    name: &str,
    value: u32,
) -> Result<()> {
    let scalar_shape: Vec<u64> = vec![];
    let path = format!("/info/{name}");

    let field = ArrayBuilder::new(scalar_shape.clone(), scalar_shape, uint32(), 0u32)
        .build(store.clone(), &path)
        .store_err_with(|| format!("failed to build {path} array"))?;
    field
        .store_metadata()
        .store_err_with(|| format!("failed to store {path} metadata"))?;
    field
        .store_chunk(&[], vec![value])
        .store_err_with(|| format!("failed to store {path} chunk"))?;
    Ok(())
}

/// Writes `root_embeddings`, the root node's representatives, to
/// `index_root/embeddings` as `dtype`.
pub fn write_index_root(
    store: &ReadableWritableListableStorage,
    root_embeddings: &Array2<f32>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) -> Result<()> {
    let shape = vec![
        root_embeddings.nrows() as u64,
        root_embeddings.ncols() as u64,
    ];
    let subset = ArraySubset::new_with_ranges(&[0..shape[0], 0..shape[1]]);
    let path = "/index_root/embeddings";
    let array = build_embeddings_array(store, path, shape, chunk_shape, dtype)?;
    store_embeddings_subset(&array, &subset, root_embeddings, dtype, path)
}

/// Appends `embeddings` and their `children` ids to the node at `group_path`,
/// creating its arrays on the first call; the ids go in `child_key`
/// (`node_ids` or `item_ids`). `border` is left at its fill value, never populated.
pub fn append_node_batch(
    store: &ReadableWritableListableStorage,
    group_path: &str,
    child_key: &str,
    embeddings: &Array2<f32>,
    children: &Array1<u32>,
    chunk_shape: &[u64],
    dtype: EmbeddingDtype,
) -> Result<()> {
    let embeddings_path = format!("{group_path}/embeddings");
    let children_path = format!("{group_path}/{child_key}");
    let is_new = matches!(
        Array::open(store.clone(), &embeddings_path),
        Err(ArrayCreateError::MissingMetadata)
    );

    zarrs_append(
        store,
        &embeddings_path,
        &children_path,
        embeddings,
        children,
        chunk_shape,
        dtype,
    )?;

    if is_new {
        let border_shape = vec![2u64];
        let border_array = ArrayBuilder::new(border_shape.clone(), border_shape, float32(), 0.0f32)
            .build(store.clone(), &format!("{group_path}/border"))
            .store_err("failed to build border array")?;
        border_array
            .store_metadata()
            .store_err("failed to store border metadata")?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "utests/writer.rs"]
mod tests;
