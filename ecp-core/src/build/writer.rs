use std::sync::Arc;

use half::f16;
use ndarray::{Array1, Array2};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::data_type::{float16, float32, uint32};
use zarrs::array::{Array, ArrayBuilder, ArraySubset, BytesToBytesCodecTraits};
use zarrs::storage::ReadableWritableListableStorage;

use crate::utils::EmbeddingDtype;

/// Without a compressor, a chunk is padded to its full declared size on
/// disk regardless of how much of it is actually written. Since a chunk is
/// sized by I/O throughput rather than expected data size (the whole point,
/// given eCP doesn't enforce cluster sizes), that padding can be the
/// difference between a few KB and tens of MB per mostly-empty node. zstd
/// compresses the fill-value padding away.
pub(super) fn compressor() -> Vec<Arc<dyn BytesToBytesCodecTraits>> {
    vec![Arc::new(ZstdCodec::new(3, false))]
}

/// Creates (on the first call for a given path) or grows and appends to a
/// paired embeddings+ids array. `dtype` sets the embeddings array's
/// precision (f32 or f16); the ids array is always uint32. `chunk_shape`
/// only applies on creation; a later call's value is ignored once the
/// array exists. Used for a node's `embeddings`/`child_key`, or the
/// representative set's `rep_embeddings`/`rep_item_ids`.
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
            array.set_shape(vec![new_vecs, dim]).expect("Failed to grow embeddings array");
            array.store_metadata().expect("Failed to store embeddings metadata");
            let subset = ArraySubset::new_with_ranges(&[existing_vecs..new_vecs, 0..dim]);
            match dtype {
                EmbeddingDtype::F32 => {
                    array.store_array_subset(&subset, embeddings).expect("Failed to append embeddings")
                }
                EmbeddingDtype::F16 => array
                    .store_array_subset(&subset, &embeddings.mapv(f16::from_f32))
                    .expect("Failed to append embeddings"),
            }

            let mut ids_array =
                Array::open(store.clone(), ids_path).expect("ids array missing alongside embeddings");
            let existing_ids = ids_array.shape()[0];
            let new_ids = existing_ids + ids.len() as u64;
            ids_array.set_shape(vec![new_ids]).expect("Failed to grow ids array");
            ids_array.store_metadata().expect("Failed to store ids metadata");
            ids_array
                .store_array_subset(&ArraySubset::new_with_ranges(&[existing_ids..new_ids]), ids)
                .expect("Failed to append ids");
        }
        Err(_) => {
            let dim = embeddings.ncols() as u64;
            let emb_shape = vec![embeddings.nrows() as u64, dim];
            let subset = ArraySubset::new_with_ranges(&[0..emb_shape[0], 0..dim]);
            match dtype {
                EmbeddingDtype::F32 => {
                    let mut emb_builder = ArrayBuilder::new(emb_shape.clone(), chunk_shape.to_vec(), float32(), 0.0f32);
                    emb_builder.bytes_to_bytes_codecs(compressor());
                    let emb_array =
                        emb_builder.build(store.clone(), embeddings_path).expect("Failed to build embeddings array");
                    emb_array.store_metadata().expect("Failed to store embeddings metadata");
                    emb_array.store_array_subset(&subset, embeddings).expect("Failed to store embeddings");
                }
                EmbeddingDtype::F16 => {
                    let mut emb_builder =
                        ArrayBuilder::new(emb_shape.clone(), chunk_shape.to_vec(), float16(), f16::from_f32(0.0));
                    emb_builder.bytes_to_bytes_codecs(compressor());
                    let emb_array =
                        emb_builder.build(store.clone(), embeddings_path).expect("Failed to build embeddings array");
                    emb_array.store_metadata().expect("Failed to store embeddings metadata");
                    emb_array
                        .store_array_subset(&subset, &embeddings.mapv(f16::from_f32))
                        .expect("Failed to store embeddings");
                }
            }

            // Same chunk-count as the embeddings array, so a given
            // chunk index lines up across both. 
            let ids_shape = vec![ids.len() as u64];
            let mut ids_builder = ArrayBuilder::new(ids_shape.clone(), vec![chunk_shape[0]], uint32(), 0u32);
            ids_builder.bytes_to_bytes_codecs(compressor());
            let ids_array = ids_builder.build(store.clone(), ids_path).expect("Failed to build ids array");
            ids_array.store_metadata().expect("Failed to store ids metadata");
            ids_array
                .store_array_subset(&ArraySubset::new_with_ranges(&[0..ids_shape[0]]), ids)
                .expect("Failed to store ids");
        }
    }
}

#[cfg(test)]
#[path = "utests/writer.rs"]
mod tests;
