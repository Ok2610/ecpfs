use pyo3::prelude::*;

use ecp_core::utils::Metric;

#[pyclass(name = "Metric", module = "ecp.metric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyMetric {
    L2,
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
