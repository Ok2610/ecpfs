use std::path::Path;
use std::sync::Arc;

use ndarray::{Array2, s};
use rust_hdf5::{DatatypeMessage, H5Dataset, H5File};
use zarrs::array::{Array, ArraySubset};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorage;

use crate::dtype::{EmbeddingDtype, dtype_of_array, read_subset_as_f32};

/// A lazily-read source of 2D embeddings: a `.h5`/`.zarr` file (stored as
/// f32, f16, uint8 or int8, `read_vecs` always returns f32) or an
/// already-resident `Array2<f32>`. Reading a range from a disk-backed source
/// doesn't load the rest of the dataset.
pub enum EmbeddingsSource {
    Hdf5(H5Dataset),
    Zarr {
        store: ReadableListableStorage,
        path: String,
    },
    /// Already resident; `read_vecs` is a plain slice, no I/O.
    Memory(Array2<f32>),
}

/// An HDF5 dataset's dtype, as the `EmbeddingDtype` it maps to.
fn hdf5_dtype(dataset: &H5Dataset) -> EmbeddingDtype {
    match dataset
        .datatype()
        .expect("Failed to read HDF5 dataset datatype")
    {
        DatatypeMessage::FloatingPoint { size: 2, .. } => EmbeddingDtype::F16,
        DatatypeMessage::FloatingPoint { size: 4, .. } => EmbeddingDtype::F32,
        DatatypeMessage::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => EmbeddingDtype::UInt8,
        DatatypeMessage::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => EmbeddingDtype::Int8,
        other => {
            panic!("unsupported embeddings dtype: {other:?} (use float32, float16, uint8 or int8)")
        }
    }
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
                EmbeddingsSource::Zarr {
                    store,
                    path: format!("/{name}"),
                }
            }
            other => {
                panic!("unsupported embeddings file format: {other:?} (use \"h5\" or \"zarr\")")
            }
        }
    }

    /// Wraps an array at `path` in an already-open store.
    pub fn from_zarr(store: ReadableListableStorage, path: String) -> Self {
        EmbeddingsSource::Zarr { store, path }
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
            EmbeddingsSource::Memory(embeddings) => (embeddings.nrows(), embeddings.ncols()),
        }
    }

    /// The vector count of one on-disk chunk, or `fallback` if the source has
    /// no chunking to align to.
    pub fn natural_chunk_vecs(&self, fallback: usize) -> usize {
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
            EmbeddingsSource::Memory(_) => fallback,
        }
    }

    /// `memory_floor_vecs` rounded up to a whole number of
    /// `natural_chunk_vecs`-sized chunks, so a batch built from the result
    /// never straddles (and double-decodes) a chunk boundary.
    pub fn chunk_aligned_batch_vecs(&self, memory_floor_vecs: usize, fallback: usize) -> usize {
        let chunk_vecs = self.natural_chunk_vecs(fallback).max(1);
        memory_floor_vecs.max(1).div_ceil(chunk_vecs) * chunk_vecs
    }

    /// The dtype embeddings are actually stored as. `Memory` always
    /// reports `F32` since it only ever holds already-upcast `Array2<f32>`
    /// data.
    pub fn native_dtype(&self) -> EmbeddingDtype {
        match self {
            EmbeddingsSource::Hdf5(dataset) => hdf5_dtype(dataset),
            EmbeddingsSource::Zarr { store, path } => {
                let array = Array::open(store.clone(), path).expect("Failed to open zarr array");
                dtype_of_array(&array, path)
            }
            EmbeddingsSource::Memory(_) => EmbeddingDtype::F32,
        }
    }

    /// Reads vecs `start..end` (all columns) as `f32`.
    pub fn read_vecs(&self, start: usize, end: usize) -> Array2<f32> {
        match self {
            EmbeddingsSource::Hdf5(dataset) => {
                let dim = dataset.shape()[1];
                let rows = end - start;
                // f16 widens to f32 inside the HDF5 crate, but it refuses to
                // read an integer dataset as f32, so integers are read at
                // their own width and widened here.
                let flat: Vec<f32> = match hdf5_dtype(dataset) {
                    EmbeddingDtype::F16 | EmbeddingDtype::F32 => dataset
                        .read_numeric_slice_as::<f32>(&[start, 0], &[rows, dim])
                        .expect("Failed to read HDF5 vec range"),
                    EmbeddingDtype::UInt8 => dataset
                        .read_numeric_slice_as::<u8>(&[start, 0], &[rows, dim])
                        .expect("Failed to read HDF5 vec range")
                        .into_iter()
                        .map(|v| v as f32)
                        .collect(),
                    EmbeddingDtype::Int8 => dataset
                        .read_numeric_slice_as::<i8>(&[start, 0], &[rows, dim])
                        .expect("Failed to read HDF5 vec range")
                        .into_iter()
                        .map(|v| v as f32)
                        .collect(),
                };
                Array2::from_shape_vec((rows, dim), flat)
                    .expect("HDF5 vec range didn't match its declared shape")
            }
            EmbeddingsSource::Zarr { store, path } => {
                let array = Array::open(store.clone(), path).expect("Failed to open zarr array");
                let dim = array.shape()[1];
                let subset = ArraySubset::new_with_ranges(&[start as u64..end as u64, 0..dim]);
                read_subset_as_f32(&array, &subset, path)
            }
            EmbeddingsSource::Memory(embeddings) => embeddings.slice(s![start..end, ..]).to_owned(),
        }
    }
}

#[cfg(test)]
#[path = "utests/source.rs"]
mod tests;
