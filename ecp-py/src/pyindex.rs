use ecp_core::search::Index;
use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::pyerror::to_pyerr;

/// An eCP index opened from disk, for searching and inserting.
///
/// ``memory_limit_bytes`` caps the index data kept in memory, nodes and open
/// queries together, and defaults to 80% of system RAM. Every method except
/// ``set_memory_limit_bytes`` releases the GIL while it works, so other Python
/// threads keep running. Use the index as a context manager, or call ``close``,
/// so unfinished queries are saved for later.
#[pyclass(module = "ecp.index")]
pub struct IndexWrapper {
    inner: Index,
    closed: bool,
}

impl IndexWrapper {
    /// Raises ValueError if the index was closed.
    fn check_not_closed(&self) -> PyResult<()> {
        if self.closed {
            Err(PyValueError::new_err("I/O operation on closed Index"))
        } else {
            Ok(())
        }
    }
}

#[pymethods]
impl IndexWrapper {
    /// Opens the index at `index_path`, keeping at most `memory_limit_bytes` of
    /// its data in memory. PyO3 drops this doc, so the class doc carries it.
    #[new]
    #[pyo3(signature = (index_path, memory_limit_bytes=ecp_core::utils::default_memory_limit_bytes()))]
    fn new(py: Python<'_>, index_path: PathBuf, memory_limit_bytes: usize) -> PyResult<Self> {
        let inner = py
            .detach(|| Index::load(index_path, Some(memory_limit_bytes)))
            .map_err(to_pyerr)?;
        Ok(IndexWrapper {
            inner,
            closed: false,
        })
    }

    /// set_memory_limit_bytes(memory_limit_bytes)
    ///
    /// Changes the memory limit without reopening the index. Lowering it below
    /// what is cached evicts right away. Meant for occasional use, not per query.
    fn set_memory_limit_bytes(&mut self, memory_limit_bytes: usize) -> PyResult<()> {
        self.check_not_closed()?;
        self.inner.set_memory_limit_bytes(Some(memory_limit_bytes));
        Ok(())
    }

    /// new_search(query, k, search_exp, max_increments, exclude_vec)
    ///
    /// Searches for ``query`` and returns ``(items, query_id)``, where ``items`` are
    /// the ``k`` best ``(score, item_id)`` pairs, lowest score first. Pass
    /// ``query_id`` to ``get_next_k_items`` for more. See :doc:`search-parameters`
    /// for the arguments and what a score means.
    fn new_search(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude_vec: Vec<u32>,
    ) -> PyResult<(Vec<(f32, u32)>, usize)> {
        self.check_not_closed()?;
        let query: Array1<f32> = query.to_owned_array();
        let exclude_set: HashSet<u32> = exclude_vec.into_iter().collect();

        let (results, query_id): (Vec<(NotNan<f32>, u32)>, usize) = py
            .detach(|| {
                self.inner
                    .new_search(query, k, search_exp, max_increments, &exclude_set)
            })
            .map_err(to_pyerr)?;

        let items: Vec<(f32, u32)> = results
            .into_iter()
            .map(|(nn, id)| (nn.into_inner(), id))
            .collect();

        Ok((items, query_id))
    }

    /// get_next_k_items(query_id, k, search_exp, max_increments, exclude_vec)
    ///
    /// Returns the next ``k`` ``(score, item_id)`` pairs of a query started with
    /// ``new_search``, searching further if needed. Empty for an unknown
    /// ``query_id``, and raises ValueError after ``close``. See
    /// :doc:`search-parameters` for the other arguments.
    fn get_next_k_items(
        &self,
        py: Python<'_>,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude_vec: Vec<u32>,
    ) -> PyResult<Vec<(f32, u32)>> {
        self.check_not_closed()?;
        let exclude_set: HashSet<u32> = exclude_vec.into_iter().collect();
        let results: Vec<(NotNan<f32>, u32)> = py
            .detach(|| {
                self.inner
                    .get_next_k_items(query_id, k, search_exp, max_increments, &exclude_set)
            })
            .map_err(to_pyerr)?;

        Ok(results
            .into_iter()
            .map(|(nn, id)| (nn.into_inner(), id))
            .collect())
    }

    /// insert(embeddings)
    ///
    /// Adds each row of ``embeddings`` to its nearest leaf and returns the new ids
    /// as ``(start_id, end_id)``, so ``range(start_id, end_id)`` lists them in row
    /// order. Rows must match the index's dimension, and be unit-length if it was
    /// built with ``is_normalized``. Leaves only grow; the tree is never rebalanced.
    fn insert(&self, py: Python<'_>, embeddings: PyReadonlyArray2<f32>) -> PyResult<(u32, u32)> {
        self.check_not_closed()?;
        let embeddings: Array2<f32> = embeddings.to_owned_array();
        let range = py
            .detach(|| self.inner.insert(embeddings))
            .map_err(to_pyerr)?;
        Ok((range.start, range.end))
    }

    /// cleanup_persisted_queries_older_than(cutoff_unix_secs)
    ///
    /// Erases every query saved to disk before ``cutoff_unix_secs``, a Unix
    /// timestamp such as ``datetime(...).timestamp()``. Returns how many were erased.
    fn cleanup_persisted_queries_older_than(
        &self,
        py: Python<'_>,
        cutoff_unix_secs: f64,
    ) -> PyResult<usize> {
        self.check_not_closed()?;
        py.detach(|| {
            self.inner
                .cleanup_persisted_queries_older_than(cutoff_unix_secs as u64)
        })
        .map_err(to_pyerr)
    }

    /// close()
    ///
    /// Saves every open query to disk so a later ``Index`` can resume it, then
    /// closes this one, raising OSError if a save failed. Any other method
    /// raises ValueError afterwards. Safe to call more than once.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        py.detach(|| self.inner.shutdown()).map_err(to_pyerr)
    }

    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc_value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.close(py)
    }
}
