use super::*;
use crate::build::writer::{build_embeddings_array, store_embeddings_subset};
use crate::test_fixtures::{as_readable_listable, as_readable_writable_listable, new_memory_store};
use ndarray::array;
use zarrs::array::ArrayBuilder;
use zarrs::array::data_type::float32;

/// Writes `embeddings` to `/embeddings` as a single f32 chunk.
fn write_embeddings(
    store: &std::sync::Arc<zarrs::storage::store::MemoryStore>,
    embeddings: &Array2<f32>,
) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    write_embeddings_with_chunk_shape(store, embeddings, &shape);
}

/// Writes `embeddings` to `/embeddings` as f32, split into chunks of
/// `chunk_shape`.
fn write_embeddings_with_chunk_shape(
    store: &std::sync::Arc<zarrs::storage::store::MemoryStore>,
    embeddings: &Array2<f32>,
    chunk_shape: &[u64],
) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape, chunk_shape.to_vec(), float32(), 0.0f32)
        .build(store.clone(), "/embeddings")
        .expect("failed to build embeddings array");
    array
        .store_metadata()
        .expect("failed to store embeddings metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[
                0..embeddings.nrows() as u64,
                0..embeddings.ncols() as u64,
            ]),
            embeddings,
        )
        .expect("failed to store embeddings");
}

/// Writes `embeddings` to `/embeddings` stored as `dtype`, the same way a
/// build does, and returns a source over it.
fn write_embeddings_as(
    store: &std::sync::Arc<zarrs::storage::store::MemoryStore>,
    embeddings: &Array2<f32>,
    dtype: EmbeddingDtype,
) -> EmbeddingsSource {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let subset = zarrs::array::ArraySubset::new_with_ranges(&[0..shape[0], 0..shape[1]]);
    let array = build_embeddings_array(
        &as_readable_writable_listable(store),
        "/embeddings",
        shape.clone(),
        &shape,
        dtype,
    )
    .unwrap();
    store_embeddings_subset(&array, &subset, embeddings, dtype, "/embeddings").unwrap();
    EmbeddingsSource::from_zarr(as_readable_listable(store), "/embeddings".to_string())
}

#[test]
fn zarr_source_reports_shape_and_reads_vector_ranges() {
    let store = new_memory_store();
    write_embeddings(
        &store,
        &array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
    );

    let source = EmbeddingsSource::Zarr {
        store: as_readable_listable(&store),
        path: "/embeddings".to_string(),
    };

    assert_eq!(source.shape().unwrap(), (4, 2));

    let vecs = source.read_vecs(1, 3).unwrap();
    assert_eq!(vecs, array![[2.0f32, 3.0], [4.0, 5.0]]);
}

#[test]
fn zarr_source_reports_its_actual_on_disk_chunk_vector_count() {
    let store = new_memory_store();
    write_embeddings_with_chunk_shape(
        &store,
        &array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        &[2, 2],
    );

    let source = EmbeddingsSource::Zarr {
        store: as_readable_listable(&store),
        path: "/embeddings".to_string(),
    };

    assert_eq!(
        source.natural_chunk_vecs(999).unwrap(),
        2,
        "fallback must be ignored when the source is chunked"
    );
}

#[test]
fn memory_source_reports_shape_and_reads_vector_ranges() {
    let source = EmbeddingsSource::Memory(array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0]]);

    assert_eq!(source.shape().unwrap(), (3, 2));
    assert_eq!(
        source.read_vecs(1, 3).unwrap(),
        array![[2.0f32, 3.0], [4.0, 5.0]]
    );
}

#[test]
fn memory_source_natural_chunk_vecs_is_always_the_fallback() {
    let source = EmbeddingsSource::Memory(array![[0.0f32, 1.0]]);

    assert_eq!(source.natural_chunk_vecs(7).unwrap(), 7);
}

#[test]
fn chunk_aligned_batch_vecs_rounds_the_memory_floor_up_to_a_whole_chunk() {
    let source = EmbeddingsSource::Memory(array![[0.0f32, 1.0]]);

    assert_eq!(
        source.chunk_aligned_batch_vecs(1, 1000).unwrap(),
        1000,
        "a floor below one chunk still returns a whole chunk"
    );
    assert_eq!(
        source.chunk_aligned_batch_vecs(1000, 1000).unwrap(),
        1000,
        "an exact multiple stays unchanged"
    );
    assert_eq!(
        source.chunk_aligned_batch_vecs(1001, 1000).unwrap(),
        2000,
        "any excess over a whole chunk rounds up to the next one"
    );
    assert_eq!(source.chunk_aligned_batch_vecs(2500, 1000).unwrap(), 3000);
}

#[test]
fn zarr_f16_source_reports_its_native_dtype() {
    let store = new_memory_store();
    let source = write_embeddings_as(&store, &array![[0.0f32, 1.0]], EmbeddingDtype::F16);

    assert_eq!(source.native_dtype().unwrap(), EmbeddingDtype::F16);
}

/// SIFT-style descriptors, stored as uint8, read back as f32 with no
/// rounding, since f32 represents every integer up to 2^24 exactly.
#[test]
fn zarr_uint8_source_reports_its_dtype_and_reads_back_exactly() {
    let store = new_memory_store();
    let embeddings = array![[0.0f32, 255.0], [1.0, 128.0]];
    let source = write_embeddings_as(&store, &embeddings, EmbeddingDtype::UInt8);

    assert_eq!(source.native_dtype().unwrap(), EmbeddingDtype::UInt8);
    assert_eq!(source.read_vecs(0, 2).unwrap(), embeddings);
}

#[test]
fn zarr_int8_source_reports_its_dtype_and_reads_back_exactly() {
    let store = new_memory_store();
    let embeddings = array![[-128.0f32, 127.0], [-1.0, 0.0]];
    let source = write_embeddings_as(&store, &embeddings, EmbeddingDtype::Int8);

    assert_eq!(source.native_dtype().unwrap(), EmbeddingDtype::Int8);
    assert_eq!(source.read_vecs(0, 2).unwrap(), embeddings);
}

#[test]
fn memory_source_native_dtype_is_always_f32() {
    let source = EmbeddingsSource::Memory(array![[0.0f32, 1.0]]);

    assert_eq!(
        source.native_dtype().unwrap(),
        crate::dtype::EmbeddingDtype::F32
    );
}

/// Makes a source over an empty int32 array, a dtype ecpfs doesn't support.
fn int32_source() -> (
    std::sync::Arc<zarrs::storage::store::MemoryStore>,
    EmbeddingsSource,
) {
    let store = new_memory_store();
    let shape = vec![2u64, 2];
    let array = ArrayBuilder::new(shape.clone(), shape, zarrs::array::data_type::int32(), 0i32)
        .build(store.clone(), "/embeddings")
        .expect("failed to build embeddings array");
    array
        .store_metadata()
        .expect("failed to store embeddings metadata");

    let source = EmbeddingsSource::Zarr {
        store: as_readable_listable(&store),
        path: "/embeddings".to_string(),
    };
    (store, source)
}

#[test]
fn zarr_native_dtype_rejects_an_unsupported_dtype() {
    let (_store, source) = int32_source();
    let err = source.native_dtype().unwrap_err();
    assert!(matches!(err, EcpError::InvalidInput(_)), "{err:?}");
}

#[test]
fn zarr_read_vecs_rejects_an_unsupported_dtype() {
    let (_store, source) = int32_source();
    let err = source.read_vecs(0, 2).unwrap_err();
    assert!(matches!(err, EcpError::InvalidInput(_)), "{err:?}");
}
