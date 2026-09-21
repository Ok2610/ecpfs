use ecp_core::EcpError;
use pyo3::PyErr;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyRuntimeError, PyValueError};

/// Maps an `EcpError` to the matching Python exception.
pub(crate) fn to_pyerr(err: EcpError) -> PyErr {
    let message = err.to_string();
    match err {
        EcpError::InvalidInput(_) => PyValueError::new_err(message),
        EcpError::NotFound(_) => PyFileNotFoundError::new_err(message),
        EcpError::Store(_) => PyOSError::new_err(message),
        EcpError::Corrupt(_) | EcpError::Usage(_) => PyRuntimeError::new_err(message),
    }
}
