use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

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

#[pymodule]
fn ecp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    m.add_class::<BuilderWrapper>()?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyEmbeddingDtype>()?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}
