// import the PyO3 machinery
use pyo3::prelude::*;
// bring in the PyModule type
use pyo3::types::PyModule;
use pyo3::Bound;

// pull in the items from your pyindex module
mod pybuilder;
mod pyindex;
mod pymetric;
use pybuilder::BuilderWrapper;
use pyindex::IndexWrapper;
use pymetric::PyMetric;

/// This is the Python extension entry point.  The name *must* match your
/// `lib.name = "ecp"` in Cargo.toml so that
/// `import ecp` works in Python.
#[pymodule]
fn ecp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    m.add_class::<BuilderWrapper>()?;
    m.add_class::<PyMetric>()?;
    Ok(())
}
