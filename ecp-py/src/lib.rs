use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod pybuilder;
mod pydtype;
mod pyindex;
mod pylogging;
mod pymetric;
use pybuilder::BuilderWrapper;
use pydtype::PyEmbeddingDtype;
use pyindex::IndexWrapper;
use pylogging::init_logging;
use pymetric::PyMetric;

// Every I/O-bound method in this module releases the GIL via py.detach.
#[pymodule(gil_used = true)]
fn ecp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    m.add_class::<BuilderWrapper>()?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyEmbeddingDtype>()?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}
