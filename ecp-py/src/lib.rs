use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod pybuilder;
mod pydtype;
mod pyerror;
mod pyindex;
mod pylogging;
mod pymetric;
use pybuilder::BuilderWrapper;
use pydtype::PyEmbeddingDtype;
use pyindex::IndexWrapper;
use pylogging::init_logging;
use pymetric::PyMetric;

// Every method in this module that performs I/O releases the GIL via py.detach.
// Nothing relies on the GIL: Index is thread-safe and PyO3 guards each &mut self call.
/// Registers the classes and functions of the `ecp` Python module.
#[pymodule(gil_used = false)]
fn ecp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    m.add_class::<BuilderWrapper>()?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyEmbeddingDtype>()?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}
