use pyo3::prelude::*;

use ecp_core::utils::Metric;

/// Distance metric used to build and search an index.
#[pyclass(name = "Metric", module = "ecp.metric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyMetric {
    /// Euclidean (L2) distance. Lower is more similar.
    L2,
    /// Inner product. Higher is more similar.
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
