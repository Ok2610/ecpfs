// bring in your other Rust modules as before
pub mod ecp_node;
pub mod ecp_index;
mod utils;

// import the PyO3 machinery
use pyo3::prelude::*;
// bring in the PyModule type
use pyo3::types::PyModule;
use pyo3::Bound;

// pull in the items from your pyindex module
mod pyindex;
use pyindex::IndexWrapper;

/// This is the Python extension entry point.  The name *must* match your
/// `lib.name = "ecp_index_rs"` in Cargo.toml so that
/// `import ecp_index_rs` works in Python.
#[pymodule]
fn ecp_index_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    Ok(())
}