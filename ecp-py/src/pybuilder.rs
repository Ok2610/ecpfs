use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use ecp_core::build::builder::Builder;
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;

use crate::pydtype::PyEmbeddingDtype;
use crate::pymetric::PyMetric;

/// Builds a new eCP index: select representatives, then build the tree.
#[pyclass(module = "ecp.builder")]
pub struct BuilderWrapper {
    inner: Builder,
}

fn parse_strategy(strategy: &str) -> PyResult<RepresentativeStrategy> {
    match strategy {
        "offset" => Ok(RepresentativeStrategy::Offset),
        "random" => Ok(RepresentativeStrategy::Random),
        other => Err(PyValueError::new_err(format!(
            "unknown strategy {other:?} (use \"offset\" or \"random\")"
        ))),
    }
}

#[pymethods]
impl BuilderWrapper {
    /// __new__(index_path, levels, metric, is_normalized=False, memory_limit_bytes=<80% of system RAM>, embedding_dtype=None, max_chunk_bytes=50 MiB)
    ///
    /// Creates a fresh index at `index_path` and returns a Builder ready to
    /// build into it. memory_limit_bytes is the memory budget for the build
    /// process (not strictly enforced). embedding_dtype of None matches
    /// each source's own dtype; forcing F16 against an f32 source
    /// downcasts real precision and logs a warning. max_chunk_bytes is the
    /// max size for one on-disk chunk. Releases the GIL for the actual
    /// work, same as every other method on this class.
    #[new]
    #[pyo3(signature = (
        index_path,
        levels,
        metric,
        is_normalized=false,
        memory_limit_bytes=ecp_core::utils::default_memory_limit_bytes(),
        embedding_dtype=None,
        max_chunk_bytes=ecp_core::build::builder::DEFAULT_MAX_CHUNK_BYTES,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        index_path: PathBuf,
        levels: u32,
        metric: PyMetric,
        is_normalized: bool,
        memory_limit_bytes: usize,
        embedding_dtype: Option<PyEmbeddingDtype>,
        max_chunk_bytes: usize,
    ) -> Self {
        let metric = metric.into();
        let embedding_dtype = embedding_dtype.map(Into::into);
        let inner = py.detach(|| {
            Builder::create(
                &index_path,
                levels,
                metric,
                is_normalized,
                memory_limit_bytes,
                embedding_dtype,
                max_chunk_bytes,
            )
        });
        BuilderWrapper { inner }
    }

    /// select_representatives(embeddings_file, target_cluster_items, strategy, fallback_batch_rows, grp_name="embeddings")
    ///
    /// Picks cluster representatives out of `embeddings_file` via `strategy`
    /// ("offset" | "random") and persists them. Must be called (or
    /// select_representatives_custom) before build.
    #[pyo3(signature = (embeddings_file, target_cluster_items, strategy, fallback_batch_rows, grp_name="embeddings"))]
    fn select_representatives(
        &mut self,
        py: Python<'_>,
        embeddings_file: PathBuf,
        target_cluster_items: usize,
        strategy: &str,
        fallback_batch_rows: usize,
        grp_name: &str,
    ) -> PyResult<()> {
        let strategy = parse_strategy(strategy)?;
        let grp_name = grp_name.to_string();
        py.detach(|| {
            let source = EmbeddingsSource::open(&embeddings_file, &grp_name);
            self.inner.select_representatives(
                &source,
                target_cluster_items,
                strategy,
                fallback_batch_rows,
            );
        });
        Ok(())
    }

    /// select_representatives_custom(ids: np.ndarray[u32, 1], embeddings: np.ndarray[f32, 2])
    ///
    /// Uses caller-supplied representatives directly instead of running a
    /// selection strategy, for when representatives come from an external
    /// clustering step. Must be called (or select_representatives) before
    /// build.
    fn select_representatives_custom(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<u32>,
        embeddings: PyReadonlyArray2<f32>,
    ) {
        let ids = ids.to_owned_array();
        let embeddings = embeddings.to_owned_array();
        py.detach(|| self.inner.select_representatives_custom(ids, embeddings));
    }

    /// build(embeddings_file, fallback_batch_rows, grp_name="embeddings")
    ///
    /// Writes the index root and descends the full tree over
    /// `embeddings_file`, using the representatives already selected via
    /// select_representatives or select_representatives_custom.
    #[pyo3(signature = (embeddings_file, fallback_batch_rows, grp_name="embeddings"))]
    fn build(
        &mut self,
        py: Python<'_>,
        embeddings_file: PathBuf,
        fallback_batch_rows: usize,
        grp_name: &str,
    ) {
        let grp_name = grp_name.to_string();
        py.detach(|| {
            let dataset = EmbeddingsSource::open(&embeddings_file, &grp_name);
            self.inner.build(&dataset, fallback_batch_rows);
        });
    }
}
