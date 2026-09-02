use super::*;
use crate::test_fixtures::{as_readable_listable, new_memory_store};
use ndarray::array;
use zarrs::array::ArrayBuilder;

fn write_embeddings(store: &std::sync::Arc<zarrs::storage::store::MemoryStore>, embeddings: &Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, float32(), 0.0f32)
        .build(store.clone(), "/embeddings")
        .expect("failed to build embeddings array");
    array.store_metadata().expect("failed to store embeddings metadata");
    array.store_chunk(&[0, 0], embeddings).expect("failed to store embeddings chunk");
}

#[test]
fn zarr_source_reports_shape_and_reads_row_ranges() {
    let store = new_memory_store();
    write_embeddings(&store, &array![[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]);

    let source = EmbeddingsSource::Zarr {
        store: as_readable_listable(&store),
        path: "/embeddings".to_string(),
    };

    assert_eq!(source.shape(), (4, 2));

    let rows = source.read_rows(1, 3);
    assert_eq!(rows, array![[2.0f32, 3.0], [4.0, 5.0]]);
}
