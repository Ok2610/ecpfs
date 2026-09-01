// import the PyO3 machinery
use pyo3::prelude::*;
// bring in the PyModule type
use pyo3::types::PyModule;
use pyo3::Bound;

// pull in the items from your pyindex module
mod pyindex;
use pyindex::IndexWrapper;

/// This is the Python extension entry point.  The name *must* match your
/// `lib.name = "engine"` in Cargo.toml so that
/// `import engine` works in Python.
#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    Ok(())
}
