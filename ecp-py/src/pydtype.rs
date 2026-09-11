use pyo3::prelude::*;

use ecp_core::utils::EmbeddingDtype;

/// On-disk embedding precision: half (F16) or full (F32) width float.
#[pyclass(
    name = "EmbeddingDtype",
    module = "ecp.dtype",
    eq,
    eq_int,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyEmbeddingDtype {
    F16,
    F32,
}

impl From<PyEmbeddingDtype> for EmbeddingDtype {
    fn from(dtype: PyEmbeddingDtype) -> Self {
        match dtype {
            PyEmbeddingDtype::F16 => EmbeddingDtype::F16,
            PyEmbeddingDtype::F32 => EmbeddingDtype::F32,
        }
    }
}
