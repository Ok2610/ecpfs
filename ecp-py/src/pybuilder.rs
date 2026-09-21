use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use ecp_core::build::builder::{Builder, ChunkSizes};
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;

use crate::pydtype::PyEmbeddingDtype;
use crate::pyerror::to_pyerr;
use crate::pymetric::PyMetric;

/// Builds a new eCP index in three steps: create the ``Builder``, call
/// ``select_representatives`` or ``select_representatives_custom``, then call
/// ``build``. Every method releases the GIL while it works.
///
/// The index is created at ``index_path``. ``levels`` is the number of node
/// levels below the root, and ``is_normalized`` should be set only if every
/// embedding is unit-length. ``memory_limit_bytes`` is a target for the build's
/// memory use rather than a hard cap, and defaults to 80% of system RAM. Leave
/// ``embedding_dtype`` as None to store each file's own dtype; a narrower one
/// logs a warning. ``rep_chunk_bytes`` is the chunk size for the representative
/// arrays and ``node_chunk_bytes`` the chunk size for the tree nodes. Measure
/// zarr read speed at a few chunk sizes on your own data before changing them.
#[pyclass(module = "ecp.builder")]
pub struct BuilderWrapper {
    inner: Builder,
}

/// Parses "offset" or "random", raising ValueError for anything else.
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
    /// Creates the index at `index_path` and returns a builder for it. PyO3
    /// drops this doc, so the class doc carries the arguments.
    #[new]
    #[pyo3(signature = (
        index_path,
        levels,
        metric,
        is_normalized=false,
        memory_limit_bytes=ecp_core::utils::default_memory_limit_bytes(),
        embedding_dtype=None,
        rep_chunk_bytes=ecp_core::build::builder::DEFAULT_REP_CHUNK_BYTES,
        node_chunk_bytes=ecp_core::build::builder::DEFAULT_NODE_CHUNK_BYTES,
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
        rep_chunk_bytes: usize,
        node_chunk_bytes: usize,
    ) -> PyResult<Self> {
        let metric = metric.into();
        let embedding_dtype = embedding_dtype.map(Into::into);
        let inner = py
            .detach(|| {
                Builder::create(
                    &index_path,
                    levels,
                    metric,
                    is_normalized,
                    memory_limit_bytes,
                    embedding_dtype,
                    ChunkSizes {
                        rep_chunk_bytes,
                        node_chunk_bytes,
                    },
                )
            })
            .map_err(to_pyerr)?;
        Ok(BuilderWrapper { inner })
    }

    /// select_representatives(embeddings_file, target_cluster_items, strategy, fallback_batch_rows, grp_name="embeddings")
    ///
    /// Picks the representatives the tree is built from, out of the dataset
    /// ``grp_name`` in ``embeddings_file`` (``.h5`` or ``.zarr``). ``strategy`` is
    /// "offset" (evenly spaced items) or "random". ``target_cluster_items`` is the
    /// average cluster size to aim for, and ``fallback_batch_rows`` the chunk size
    /// assumed if the file isn't chunked.
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
            let source = EmbeddingsSource::open(&embeddings_file, &grp_name)?;
            self.inner.select_representatives(
                &source,
                target_cluster_items,
                strategy,
                fallback_batch_rows,
            )
        })
        .map_err(to_pyerr)
    }

    /// select_representatives_custom(ids, embeddings)
    ///
    /// Uses your own representatives instead of picking them, such as ones from
    /// an external clustering step. ``ids[i]`` is the id of row ``i`` of
    /// ``embeddings``.
    fn select_representatives_custom(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<u32>,
        embeddings: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        let ids = ids.to_owned_array();
        let embeddings = embeddings.to_owned_array();
        py.detach(|| self.inner.select_representatives_custom(ids, embeddings))
            .map_err(to_pyerr)
    }

    /// build(embeddings_file, fallback_batch_rows, grp_name="embeddings")
    ///
    /// Builds the tree from the dataset ``grp_name`` in ``embeddings_file``, whose
    /// rows get item ids 0, 1, 2, ... in order. ``fallback_batch_rows`` works as
    /// in ``select_representatives``.
    #[pyo3(signature = (embeddings_file, fallback_batch_rows, grp_name="embeddings"))]
    fn build(
        &mut self,
        py: Python<'_>,
        embeddings_file: PathBuf,
        fallback_batch_rows: usize,
        grp_name: &str,
    ) -> PyResult<()> {
        let grp_name = grp_name.to_string();
        py.detach(|| {
            let dataset = EmbeddingsSource::open(&embeddings_file, &grp_name)?;
            self.inner.build(&dataset, fallback_batch_rows)
        })
        .map_err(to_pyerr)
    }
}
