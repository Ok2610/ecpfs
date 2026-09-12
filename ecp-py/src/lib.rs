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
// support. IndexWrapper's concurrent-hot-path methods (new_search,
// get_next_k_items, insert) are interior-mutable and release the GIL
// already, so they don't rely on it for correctness. BuilderWrapper is
// unaudited: its methods still hold the GIL for their whole duration.
#[pymodule(gil_used = true)]
fn ecp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWrapper>()?;
    m.add_class::<BuilderWrapper>()?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyEmbeddingDtype>()?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}
