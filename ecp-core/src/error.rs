//! ecp-core's error type. Every fallible function in this crate returns
//! [`Result`], `std::result::Result<T, EcpError>`.

use std::fmt;

use thiserror::Error;

/// Why an ecp-core call failed. Each variant maps to one Python exception in
/// ecp-py.
#[derive(Debug, Clone, Error)]
pub enum EcpError {
    /// A bad argument, or bad data in the user's file: a dimension mismatch,
    /// an unsupported dtype or file format, an unknown metric name,
    /// `levels = 0`, a dimension too wide for one chunk, or ids and
    /// embeddings of different lengths. Maps to Python's `ValueError`.
    #[error("{0}")]
    InvalidInput(String),
    /// The embeddings file or the index directory is not there. Maps to
    /// Python's `FileNotFoundError`.
    #[error("{0}")]
    NotFound(String),
    /// The store failed to read, write or erase: permissions, a full disk, a
    /// truncated chunk. Maps to Python's `OSError`.
    #[error("{0}")]
    Store(String),
    /// The index is readable but inconsistent: a missing `info` field, a node
    /// with embeddings but no ids, an unparsable saved query. Maps to
    /// Python's `RuntimeError`.
    #[error("{0}")]
    Corrupt(String),
    /// The API was called in the wrong order, such as `build` before
    /// `select_representatives`. Maps to Python's `RuntimeError`.
    #[error("{0}")]
    Usage(String),
}

/// This crate's result type.
pub type Result<T> = std::result::Result<T, EcpError>;

/// Turns a lower layer's `Result<T, E>` into ecp-core's [`Result`], prefixing
/// `context` to the error's message. Every call site here is a store read,
/// write or erase failing, so it always maps to [`EcpError::Store`].
pub(crate) trait ResultExt<T> {
    fn store_err(self, context: &str) -> Result<T>;
    /// Like `store_err`, but builds the context only if the call fails.
    fn store_err_with(self, context: impl FnOnce() -> String) -> Result<T>;
}

impl<T, E: fmt::Display> ResultExt<T> for std::result::Result<T, E> {
    fn store_err(self, context: &str) -> Result<T> {
        self.map_err(|e| EcpError::Store(format!("{context}: {e}")))
    }

    fn store_err_with(self, context: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|e| EcpError::Store(format!("{}: {e}", context())))
    }
}

#[cfg(test)]
#[path = "utests/error.rs"]
mod tests;
