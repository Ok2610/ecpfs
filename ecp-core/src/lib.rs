//! The core library of ecpfs. It builds and searches Extended Cluster Pruning
//! (eCP) indexes, stored as zarr arrays on disk.
//!
//! - [`build::builder::Builder`] builds an index from embeddings in an `.h5` or `.zarr` file.
//! - [`search::Index`] opens an index for k-nearest-neighbor search and inserts.
//! - [`search::IndexInfo`] reads an index's settings without loading its tree.
//! - [`logging::init`] turns on logging to a JSONL file.
//! - [`error::EcpError`] is what a fallible call in this crate returns on failure.
//! - [`format::FORMAT_VERSION`] is the on-disk format version this build writes and reads.
//!
//! The `ecpfs` Python package and the `ecp` command line tool both wrap this crate.

pub mod build;
pub mod dtype;
pub mod error;
pub mod format;
pub mod logging;
pub mod metric;
pub mod search;
#[cfg(test)]
mod test_fixtures;
pub mod utils;

pub use error::{EcpError, Result};
