use half::f16;
use ndarray::Array2;
use zarrs::array::data_type::{float16, float32, int8, uint8};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

/// On-disk width for embeddings arrays. Narrower than `F32` means less disk
/// and less to read, but nothing is cached narrow: every read widens to f32
/// (see `read_subset_as_f32`), so resident size is the same whichever is
/// chosen. `F16` loses precision on genuinely `F32` data, while `UInt8`
/// (`0..=255`, SIFT-style descriptors) and `Int8` (`-128..=127`, symmetric
/// scalar quantization) round-trip integer-valued data exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDtype {
    UInt8,
    Int8,
    F16,
    F32,
}

impl EmbeddingDtype {
    /// True if storing as `self` cannot represent every value `native` can,
    /// so writing narrows the data. `Int8` and `UInt8` are each lossy for
    /// the other: neither range contains the other's.
    pub fn narrows(self, native: EmbeddingDtype) -> bool {
        use EmbeddingDtype::{F16, F32, Int8, UInt8};
        matches!(
            (self, native),
            (F16, F32)
                | (UInt8, F32)
                | (UInt8, F16)
                | (UInt8, Int8)
                | (Int8, F32)
                | (Int8, F16)
                | (Int8, UInt8)
        )
    }
}

/// The dtype `array`'s elements are stored as. `context` names the array in
/// the panic message when it holds a dtype ecpfs can't read.
pub fn dtype_of_array<T: ?Sized>(array: &Array<T>, context: &str) -> EmbeddingDtype {
    let dtype = array.data_type();
    if *dtype == float32() {
        EmbeddingDtype::F32
    } else if *dtype == float16() {
        EmbeddingDtype::F16
    } else if *dtype == uint8() {
        EmbeddingDtype::UInt8
    } else if *dtype == int8() {
        EmbeddingDtype::Int8
    } else {
        panic!(
            "unsupported embeddings dtype: {context} is {dtype:?} (use float32, float16, uint8 or int8)"
        )
    }
}

/// Reads `subset` of `array` as f32, widening from whatever dtype it's
/// stored as. Every distance computation runs on f32, so this is the one
/// place a stored dtype is widened on the read path.
pub fn read_subset_as_f32<T: ReadableStorageTraits + ?Sized + 'static>(
    array: &Array<T>,
    subset: &ArraySubset,
    context: &str,
) -> Array2<f32> {
    match dtype_of_array(array, context) {
        EmbeddingDtype::F32 => array
            .retrieve_array_subset::<Array2<f32>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}")),
        EmbeddingDtype::F16 => array
            .retrieve_array_subset::<Array2<f16>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}"))
            .mapv(|x| x.to_f32()),
        EmbeddingDtype::UInt8 => array
            .retrieve_array_subset::<Array2<u8>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}"))
            .mapv(|x| x as f32),
        EmbeddingDtype::Int8 => array
            .retrieve_array_subset::<Array2<i8>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}"))
            .mapv(|x| x as f32),
    }
}

#[cfg(test)]
#[path = "utests/dtype.rs"]
mod tests;
