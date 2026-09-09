use pyo3::prelude::*;

use ecp_core::utils::EmbeddingDtype;

/// On-disk embedding precision. `None` (the default everywhere this is
/// used) matches the source's own dtype; forcing `F16` against an `f32`
/// source downcasts real precision and logs a warning.
#[pyclass(name = "EmbeddingDtype", module = "ecp.dtype", eq, eq_int, from_py_object)]
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
