//! Exercises `EmbeddingsSource::Hdf5` against a real `.h5` file, since
//! `rust-hdf5` has no in-memory driver to test against (unlike the `Zarr`
//! variant, covered by a fast `MemoryStore`-backed unit test).

use rust_hdf5::H5File;

use ecp_core::build::source::EmbeddingsSource;

#[test]
fn hdf5_source_reports_shape_and_reads_row_ranges() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("embeddings.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    let dataset = file
        .new_dataset::<f32>()
        .shape(&[4usize, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    dataset
        .write_raw(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .expect("failed to write HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    let source = EmbeddingsSource::open(&file_path, "embeddings");

    assert_eq!(source.shape(), (4, 2));
    assert_eq!(source.natural_batch_rows(999), 999, "contiguous storage has no chunk alignment to exploit");

    let rows = source.read_rows(1, 3);
    assert_eq!(rows, ndarray::array![[2.0f32, 3.0], [4.0, 5.0]]);
}

#[test]
fn hdf5_source_reports_its_actual_on_disk_chunk_row_count() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("chunked.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    file.new_dataset::<f32>()
        .shape(&[4usize, 2])
        .chunk(&[2, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    let source = EmbeddingsSource::open(&file_path, "embeddings");

    assert_eq!(source.natural_batch_rows(999), 2, "fallback must be ignored when the source is chunked");
}
