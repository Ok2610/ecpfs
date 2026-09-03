use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use std::path::PathBuf;

use ecp_core::build::builder::Builder;
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;

use crate::pymetric::PyMetric;

#[pyclass(module = "ecp.builder")]
pub struct BuilderWrapper {
    inner: Builder,
}

fn parse_strategy(strategy: &str) -> PyResult<RepresentativeStrategy> {
    match strategy {
        "offset" => Ok(RepresentativeStrategy::Offset),
        "random" => Ok(RepresentativeStrategy::Random),
        other => Err(PyValueError::new_err(format!("unknown strategy {other:?} (use \"offset\" or \"random\")"))),
    }
}

#[pymethods]
impl BuilderWrapper {
    /// __new__(index_path, levels, metric, is_normalized=False, memory_limit_bytes=4 GiB)
    #[new]
    #[pyo3(signature = (index_path, levels, metric, is_normalized=false, memory_limit_bytes=4 * 1024 * 1024 * 1024))]
    fn new(index_path: PathBuf, levels: u32, metric: PyMetric, is_normalized: bool, memory_limit_bytes: usize) -> Self {
        BuilderWrapper { inner: Builder::create(&index_path, levels, metric.into(), is_normalized, memory_limit_bytes) }
    }

    /// select_representatives(embeddings_file, target_cluster_items, strategy, fallback_batch_rows, grp_name="embeddings")
    ///
    /// strategy: "offset" | "random"
    #[pyo3(signature = (embeddings_file, target_cluster_items, strategy, fallback_batch_rows, grp_name="embeddings"))]
    fn select_representatives(
        &mut self,
        embeddings_file: PathBuf,
        target_cluster_items: usize,
        strategy: &str,
        fallback_batch_rows: usize,
        grp_name: &str,
    ) -> PyResult<()> {
        let strategy = parse_strategy(strategy)?;
        let source = EmbeddingsSource::open(&embeddings_file, grp_name);
        self.inner.select_representatives(&source, target_cluster_items, strategy, fallback_batch_rows);
        Ok(())
    }

    /// select_representatives_custom(ids: np.ndarray[u32, 1], embeddings: np.ndarray[f32, 2])
    fn select_representatives_custom(&mut self, ids: PyReadonlyArray1<u32>, embeddings: PyReadonlyArray2<f32>) {
        self.inner.select_representatives_custom(ids.to_owned_array(), embeddings.to_owned_array());
    }

    /// build(embeddings_file, fallback_batch_rows, grp_name="embeddings")
    #[pyo3(signature = (embeddings_file, fallback_batch_rows, grp_name="embeddings"))]
    fn build(&mut self, embeddings_file: PathBuf, fallback_batch_rows: usize, grp_name: &str) {
        let dataset = EmbeddingsSource::open(&embeddings_file, grp_name);
        self.inner.build(&dataset, fallback_batch_rows);
    }
}
