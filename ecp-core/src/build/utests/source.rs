use super::*;
use crate::test_fixtures::{as_readable_listable, new_memory_store};
use ndarray::array;
use zarrs::array::ArrayBuilder;

fn write_embeddings(store: &std::sync::Arc<zarrs::storage::store::MemoryStore>, embeddings: &Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    write_embeddings_with_chunk_shape(store, embeddings, &shape);
}

fn write_embeddings_with_chunk_shape(
    store: &std::sync::Arc<zarrs::storage::store::MemoryStore>,
    embeddings: &Array2<f32>,
    chunk_shape: &[u64],
) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape, chunk_shape.to_vec(), float32(), 0.0f32)
        .build(store.clone(), "/embeddings")
        .expect("failed to build embeddings array");
    array.store_metadata().expect("failed to store embeddings metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[0..embeddings.nrows() as u64, 0..embeddings.ncols() as u64]),
            embeddings,
        )
        .expect("failed to store embeddings");
}

#[test]
fn zarr_source_reports_shape_and_reads_vec_ranges() {
    let store = new_memory_store();
    write_embeddings(&store, &array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]);

    let source = EmbeddingsSource::Zarr {
        store: as_readable_listable(&store),
        path: "/embeddings".to_string(),
    };

    assert_eq!(source.shape(), (4, 2));

    let vecs = source.read_vecs(1, 3);
    assert_eq!(vecs, array![[2.0f32, 3.0], [4.0, 5.0]]);
}

#[test]
fn zarr_source_reports_its_actual_on_disk_chunk_vec_count() {
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

    assert_eq!(source.natural_batch_vecs(999), 2, "fallback must be ignored when the source is chunked");
}

#[test]
fn from_zarr_reads_the_same_as_the_zarr_variant() {
    let store = new_memory_store();
    write_embeddings(&store, &array![[0.0f32, 1.0], [2.0, 3.0]]);

    let source = EmbeddingsSource::from_zarr(as_readable_listable(&store), "/embeddings".to_string());

    assert_eq!(source.shape(), (2, 2));
    assert_eq!(source.read_vecs(0, 2), array![[0.0f32, 1.0], [2.0, 3.0]]);
}

#[test]
fn memory_source_reports_shape_and_reads_vec_ranges_with_no_io() {
    let source = EmbeddingsSource::Memory(array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0]]);

    assert_eq!(source.shape(), (3, 2));
    assert_eq!(source.read_vecs(1, 3), array![[2.0f32, 3.0], [4.0, 5.0]]);
}

#[test]
fn memory_source_natural_batch_vecs_is_always_the_fallback() {
    let source = EmbeddingsSource::Memory(array![[0.0f32, 1.0]]);

    assert_eq!(source.natural_batch_vecs(7), 7);
}

#[test]
fn zarr_source_reports_its_native_dtype() {
    let store = new_memory_store();
    write_embeddings(&store, &array![[0.0f32, 1.0]]);
    let source = EmbeddingsSource::Zarr { store: as_readable_listable(&store), path: "/embeddings".to_string() };

    assert_eq!(source.native_dtype(), crate::utils::EmbeddingDtype::F32);
}

#[test]
fn zarr_f16_source_reports_its_native_dtype() {
    let store = new_memory_store();
    let embeddings = array![[0.0f32, 1.0]];
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, zarrs::array::data_type::float16(), half::f16::from_f32(0.0))
        .build(store.clone(), "/embeddings")
        .expect("failed to build embeddings array");
    array.store_metadata().expect("failed to store embeddings metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[0..embeddings.nrows() as u64, 0..embeddings.ncols() as u64]),
            &embeddings.mapv(half::f16::from_f32),
        )
        .expect("failed to store embeddings");
    let source = EmbeddingsSource::Zarr { store: as_readable_listable(&store), path: "/embeddings".to_string() };

    assert_eq!(source.native_dtype(), crate::utils::EmbeddingDtype::F16);
}

#[test]
fn memory_source_native_dtype_is_always_f32() {
    let source = EmbeddingsSource::Memory(array![[0.0f32, 1.0]]);

    assert_eq!(source.native_dtype(), crate::utils::EmbeddingDtype::F32);
}
