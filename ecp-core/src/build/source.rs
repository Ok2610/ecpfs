use std::path::Path;
use std::sync::Arc;

use half::f16;
use ndarray::Array2;
use rust_hdf5::{H5Dataset, H5File};
use zarrs::array::data_type::{float16, float32};
use zarrs::array::{Array, ArraySubset};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorage;

/// A lazily-read source of 2D f32 embeddings, opened from a `.h5` or
/// `.zarr` file. Reading a row range doesn't load the rest of the dataset.
pub enum EmbeddingsSource {
    Hdf5(H5Dataset),
    Zarr { store: ReadableListableStorage, path: String },
}

impl EmbeddingsSource {
    /// Opens `path`'s `.h5` or `.zarr` dataset/array named `name`.
    pub fn open(path: &Path, name: &str) -> Self {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("h5") => {
                let file = H5File::open(path).expect("Failed to open HDF5 file");
                let dataset = file.dataset(name).expect("Failed to open HDF5 dataset");
                EmbeddingsSource::Hdf5(dataset)
            }
            Some("zarr") => {
                let store: ReadableListableStorage =
                    Arc::new(FilesystemStore::new(path).expect("Failed to open zarr store"));
                EmbeddingsSource::Zarr { store, path: format!("/{name}") }
            }
            other => panic!("unsupported embeddings file format: {other:?} (use \"h5\" or \"zarr\")"),
        }
    }

    /// `(total_items, dim)`.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            EmbeddingsSource::Hdf5(dataset) => {
                let shape = dataset.shape();
                (shape[0], shape[1])
            }
            EmbeddingsSource::Zarr { store, path } => {
                let array = Array::open(store.clone(), path).expect("Failed to open zarr array");
                let shape = array.shape();
                (shape[0] as usize, shape[1] as usize)
            }
        }
    }

    /// The row count of one on-disk chunk, if the source is chunked -
    /// `read_rows` on a range aligned to this size touches exactly one
    /// chunk. `fallback` is used when there's no such alignment to exploit
    /// (HDF5 contiguous storage, where a partial read is already a plain
    /// byte-range read with no decompression unit to align to).
    pub fn natural_batch_rows(&self, fallback: usize) -> usize {
        match self {
            EmbeddingsSource::Hdf5(dataset) => {
                dataset.chunk_dims().map(|dims| dims[0]).unwrap_or(fallback)
            }
            EmbeddingsSource::Zarr { store, path } => {
                let array = Array::open(store.clone(), path).expect("Failed to open zarr array");
                array
                    .chunk_shape_usize(&[0, 0])
                    .expect("Failed to read zarr chunk shape")[0]
            }
        }
    }

    /// Reads rows `start..end` (all columns) as `f32`.
    pub fn read_rows(&self, start: usize, end: usize) -> Array2<f32> {
        match self {
            EmbeddingsSource::Hdf5(dataset) => {
                let dim = dataset.shape()[1];
                let flat = dataset
                    .read_slice::<f32>(&[start, 0], &[end - start, dim])
                    .expect("Failed to read HDF5 row range");
                Array2::from_shape_vec((end - start, dim), flat)
                    .expect("HDF5 row range didn't match its declared shape")
            }
            EmbeddingsSource::Zarr { store, path } => {
                let array = Array::open(store.clone(), path).expect("Failed to open zarr array");
                let dim = array.shape()[1];
                let subset = ArraySubset::new_with_ranges(&[
                    start as u64..end as u64,
                    0..dim,
                ]);
                let dtype = array.data_type();
                if *dtype != float32() && *dtype != float16() {
                    panic!("unsupported embeddings dtype: {dtype:?} (use float32 or float16)")
                }
                if *dtype == float32() {
                    array
                        .retrieve_array_subset::<Array2<f32>>(&subset)
                        .expect("Failed to read zarr row range")
                } else {
                    array
                        .retrieve_array_subset::<Array2<f16>>(&subset)
                        .expect("Failed to read zarr row range")
                        .mapv(|x| x.to_f32())
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "utests/source.rs"]
mod tests;
