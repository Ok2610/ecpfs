use std::sync::Arc;

use ndarray::{Array1, Array2};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::data_type::{float32, uint32};
use zarrs::array::{Array, ArrayBuilder, ArraySubset, BytesToBytesCodecTraits};
use zarrs::storage::ReadableWritableListableStorage;

/// Without a compressor, a chunk is padded to its full declared size on
/// disk regardless of how much of it is actually written - for a chunk
/// sized by I/O throughput rather than expected data size (the whole
/// point, given eCP doesn't enforce cluster sizes), that's the difference
/// between a few KB and tens of MB per mostly-empty node. zstd compresses
/// the fill-value padding away.
pub(super) fn compressor() -> Vec<Arc<dyn BytesToBytesCodecTraits>> {
    vec![Arc::new(ZstdCodec::new(3, false))]
}

/// Creates (on the first call for a given path) or grows and appends to
/// (on every later call) a paired embeddings+ids array - a node's
/// `embeddings`/`child_key`, or the representative set's
/// `rep_embeddings`/`rep_item_ids`. Never requires more than one batch in
/// memory. The `zarrs`-backed implementation; a future flat binary format
/// would get its own `bin_append`.
pub fn zarrs_append(
    store: &ReadableWritableListableStorage,
    embeddings_path: &str,
    ids_path: &str,
    embeddings: &Array2<f32>,
    ids: &Array1<u32>,
    chunk_shape: &[u64],
) {
    match Array::open(store.clone(), embeddings_path) {
        Ok(mut array) => {
            let existing_rows = array.shape()[0];
            let dim = array.shape()[1];
            let new_rows = existing_rows + embeddings.nrows() as u64;
            array.set_shape(vec![new_rows, dim]).expect("Failed to grow embeddings array");
            array.store_metadata().expect("Failed to store embeddings metadata");
            array
                .store_array_subset(&ArraySubset::new_with_ranges(&[existing_rows..new_rows, 0..dim]), embeddings)
                .expect("Failed to append embeddings");

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
            let mut emb_builder = ArrayBuilder::new(emb_shape.clone(), chunk_shape.to_vec(), float32(), 0.0f32);
            emb_builder.bytes_to_bytes_codecs(compressor());
            let emb_array = emb_builder
                .build(store.clone(), embeddings_path)
                .expect("Failed to build embeddings array");
            emb_array.store_metadata().expect("Failed to store embeddings metadata");
            emb_array
                .store_array_subset(&ArraySubset::new_with_ranges(&[0..emb_shape[0], 0..dim]), embeddings)
                .expect("Failed to store embeddings");

            // Same row-chunk-count as the embeddings array, so a given
            // chunk index lines up across both - not derived from this
            // first batch's size, which would lock in an arbitrary chunk
            // shape as more batches append.
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
