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

// PyO3 0.29 defaults every module to declaring free-threaded (no-GIL)
// support. IndexWrapper/BuilderWrapper's &mut self methods aren't racy
// under that (PyO3's own per-object exclusive-borrow check still applies),
// but ecp_core::Index's internal caching isn't real interior-mutability
// yet.
#[pymodule(gil_used = true)]
fn ecp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    m.add_class::<BuilderWrapper>()?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyEmbeddingDtype>()?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}
