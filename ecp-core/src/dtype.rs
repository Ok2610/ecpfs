//! The types embeddings can be stored as on disk, and reading them back as f32.

use half::f16;
use ndarray::Array2;
use zarrs::array::data_type::{float16, float32, int8, uint8};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

/// The type an index stores its embeddings as on disk. A narrower type uses
/// less disk, but every read widens to f32, so memory use is the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDtype {
    /// 8-bit unsigned integers (0 to 255), such as SIFT descriptors. Exact for
    /// integer data.
    UInt8,
    /// 8-bit signed integers (-128 to 127), such as scalar-quantized vectors.
    /// Exact for integer data.
    Int8,
    /// 16-bit floats. Loses precision on f32 data.
    F16,
    /// 32-bit floats.
    F32,
}

impl EmbeddingDtype {
    /// Checks whether storing `native` data as `self` loses information, because
    /// `self` can't hold every value `native` can. `Int8` and `UInt8` each lose
    /// information for the other, since neither range contains the other.
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

/// Returns the dtype that `array` is stored as. `context` names the array in
/// the panic message if ecpfs can't read that dtype.
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

/// Reads the `subset` region of `array` as f32, widening from the stored
/// dtype, since distances are always computed on f32. `context` names the
/// array in panic messages.
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
