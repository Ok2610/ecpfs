use pyo3::prelude::*;

use ecp_core::utils::Metric;

/// How the distance between two vectors is measured: ``L2`` (Euclidean
/// distance, lower is closer) or ``IP`` (inner product, higher is closer).
#[pyclass(name = "Metric", module = "ecp.metric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyMetric {
    /// Euclidean distance; lower is closer.
    L2,
    /// Inner product; higher is closer.
    IP,
}

impl From<PyMetric> for Metric {
    fn from(metric: PyMetric) -> Self {
        match metric {
            PyMetric::L2 => Metric::L2,
            PyMetric::IP => Metric::IP,
        }
    }
}
