//! Tests `EmbeddingsSource::Hdf5` against `.h5` files on disk, since
//! `rust-hdf5` has no in-memory driver. The `Zarr` variant is covered by
//! in-memory unit tests.

use rust_hdf5::{DatatypeMessage, H5File};

use ecp_core::EcpError;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::utils::EmbeddingDtype;

#[test]
fn hdf5_source_reports_shape_and_reads_vec_ranges() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("embeddings.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    let dataset = file
        .new_dataset::<f32>()
        .shape([4usize, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    dataset
        .write_raw(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .expect("failed to write HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    let source = EmbeddingsSource::open(&file_path, "embeddings").unwrap();

    assert_eq!(source.shape().unwrap(), (4, 2));
    assert_eq!(
        source.natural_chunk_vecs(999).unwrap(),
        999,
        "contiguous storage has no chunk alignment to exploit"
    );

    let vecs = source.read_vecs(1, 3).unwrap();
    assert_eq!(vecs, ndarray::array![[2.0f32, 3.0], [4.0, 5.0]]);
}

#[test]
fn hdf5_source_reports_its_actual_on_disk_chunk_vec_count() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("chunked.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    file.new_dataset::<f32>()
        .shape([4usize, 2])
        .chunk(&[2, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    let source = EmbeddingsSource::open(&file_path, "embeddings").unwrap();

    assert_eq!(
        source.natural_chunk_vecs(999).unwrap(),
        2,
        "fallback must be ignored when the source is chunked"
    );
}

#[test]
fn hdf5_f16_source_reads_correctly_upcast_to_f32() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("f16_embeddings.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    let dataset = file
        .new_dataset::<u16>()
        .datatype(DatatypeMessage::f16_type())
        .shape([2usize, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    let bits: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    dataset
        .write_raw(&bits)
        .expect("failed to write HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    let source = EmbeddingsSource::open(&file_path, "embeddings").unwrap();
    assert_eq!(source.native_dtype().unwrap(), EmbeddingDtype::F16);

    let vecs = source.read_vecs(0, 2).unwrap();
    assert_eq!(vecs, ndarray::array![[1.0f32, 2.0], [3.0, 4.0]]);
}

/// Writes a u32 `.h5` dataset, a dtype ecpfs doesn't support.
fn write_int_dataset() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("int_embeddings.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    file.new_dataset::<u32>()
        .shape([2usize, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    (tmp, file_path)
}

#[test]
fn hdf5_native_dtype_rejects_an_unsupported_dtype() {
    let (_tmp, file_path) = write_int_dataset();
    let source = EmbeddingsSource::open(&file_path, "embeddings").unwrap();
    let err = source.native_dtype().unwrap_err();
    assert!(matches!(err, EcpError::InvalidInput(_)), "{err:?}");
}

#[test]
fn hdf5_read_vecs_rejects_an_unsupported_dtype() {
    let (_tmp, file_path) = write_int_dataset();
    let source = EmbeddingsSource::open(&file_path, "embeddings").unwrap();
    let err = source.read_vecs(0, 2).unwrap_err();
    assert!(matches!(err, EcpError::InvalidInput(_)), "{err:?}");
}

#[test]
fn hdf5_open_reports_a_missing_dataset_as_not_found() {
    let (_tmp, file_path) = write_int_dataset();
    let err = EmbeddingsSource::open(&file_path, "no_such_dataset")
        .err()
        .expect("opening a missing dataset should fail");
    assert!(matches!(err, EcpError::NotFound(_)), "{err:?}");
}

/// SIFT descriptors are often distributed as uint8. The HDF5 crate won't
/// read an integer dataset as f32, so `read_vecs` reads it as uint8 and
/// widens it.
#[test]
fn hdf5_uint8_source_reads_correctly_widened_to_f32() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("uint8_embeddings.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    let dataset = file
        .new_dataset::<u8>()
        .shape([2usize, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    dataset
        .write_raw(&[0u8, 255, 1, 128])
        .expect("failed to write HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    let source = EmbeddingsSource::open(&file_path, "embeddings").unwrap();
    assert_eq!(source.native_dtype().unwrap(), EmbeddingDtype::UInt8);

    let vecs = source.read_vecs(0, 2).unwrap();
    assert_eq!(vecs, ndarray::array![[0.0f32, 255.0], [1.0, 128.0]]);
}

#[test]
fn hdf5_int8_source_reads_correctly_widened_to_f32() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp.path().join("int8_embeddings.h5");

    let file = H5File::create(&file_path).expect("failed to create HDF5 file");
    let dataset = file
        .new_dataset::<i8>()
        .shape([2usize, 2])
        .create("embeddings")
        .expect("failed to create HDF5 dataset");
    dataset
        .write_raw(&[-128i8, 127, -1, 0])
        .expect("failed to write HDF5 dataset");
    file.close().expect("failed to close HDF5 file");

    let source = EmbeddingsSource::open(&file_path, "embeddings").unwrap();
    assert_eq!(source.native_dtype().unwrap(), EmbeddingDtype::Int8);

    let vecs = source.read_vecs(0, 2).unwrap();
    assert_eq!(vecs, ndarray::array![[-128.0f32, 127.0], [-1.0, 0.0]]);
}
