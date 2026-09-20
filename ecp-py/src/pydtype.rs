use pyo3::prelude::*;

use ecp_core::utils::EmbeddingDtype;

/// The type an index stores its embeddings as on disk: ``F32``, ``F16``,
/// ``UInt8`` (0 to 255) or ``Int8`` (-128 to 127). Every read widens to f32,
/// so a narrower type saves disk, not memory.
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
