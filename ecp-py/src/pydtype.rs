use pyo3::prelude::*;

use ecp_core::utils::EmbeddingDtype;

/// On-disk embedding width: half (F16) or full (F32) width float, or 8-bit
/// integer, unsigned (UInt8, `0..=255`) or signed (Int8, `-128..=127`).
/// Every read widens to f32 regardless, so a narrower dtype saves disk and
/// read bandwidth, not memory.
#[pyclass(
    name = "EmbeddingDtype",
    module = "ecp.dtype",
    eq,
    eq_int,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyEmbeddingDtype {
    UInt8,
    Int8,
    F16,
    F32,
}

impl From<PyEmbeddingDtype> for EmbeddingDtype {
    fn from(dtype: PyEmbeddingDtype) -> Self {
        match dtype {
            PyEmbeddingDtype::UInt8 => EmbeddingDtype::UInt8,
            PyEmbeddingDtype::Int8 => EmbeddingDtype::Int8,
            PyEmbeddingDtype::F16 => EmbeddingDtype::F16,
            PyEmbeddingDtype::F32 => EmbeddingDtype::F32,
        }
    }
}
