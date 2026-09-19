use ecp_core::search::Index;
use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;

/// A loaded eCP index, ready to search.
#[pyclass(module = "ecp.index")]
pub struct IndexWrapper {
    inner: Index,
    closed: bool,
}

impl IndexWrapper {
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
    /// __new__(index_path: PathBuf, memory_limit_bytes: int = <80% of system RAM>)
    ///
    /// Loads an index from disk, deriving its metric and levels from the
    /// store itself. memory_limit_bytes caps how many touched nodes stay
    /// cached. Releases the GIL for the actual load.
    #[new]
    #[pyo3(signature = (index_path, memory_limit_bytes=ecp_core::utils::default_memory_limit_bytes()))]
    fn new(py: Python<'_>, index_path: PathBuf, memory_limit_bytes: usize) -> PyResult<Self> {
        let inner = py.detach(|| Index::load(index_path, Some(memory_limit_bytes)));
        Ok(IndexWrapper {
            inner,
            closed: false,
        })
    }

    /// set_memory_limit_bytes(self, memory_limit_bytes: int)
    ///
    /// Raises or lowers the memory limit on an already-loaded index, no
    /// reload needed. Lowering below what's currently resident evicts
    /// immediately. Holds the GIL for the whole call: unlike search/insert,
    /// this isn't on the concurrent hot path, so it isn't worth the extra
    /// cache-rebuild complexity to change that.
    fn set_memory_limit_bytes(&mut self, memory_limit_bytes: usize) -> PyResult<()> {
        self.check_not_closed()?;
        self.inner.set_memory_limit_bytes(Some(memory_limit_bytes));
        Ok(())
    }

    /// new_search(self, query: np.ndarray[f32, 1], k: int,
    ///            search_exp: u32, max_increments: i32, exclude_vec: list[int])
    ///
    /// Returns `(items, query_id)`, where `items: List[(score: float, item_id: int)]`.
    /// score ranks ascending (lower is better) rather than measuring a
    /// literal distance, since IP's score is a negated similarity where a
    /// strong match can be negative. Releases the GIL for the actual
    /// search, so other Python threads (including ones calling insert) run
    /// concurrently.
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

        let (results, query_id): (Vec<(NotNan<f32>, u32)>, usize) = py.detach(|| {
            self.inner
                .new_search(query, k, search_exp, max_increments, &exclude_set)
        });

        let items: Vec<(f32, u32)> = results
            .into_iter()
            .map(|(nn, id)| (nn.into_inner(), id))
            .collect();

        Ok((items, query_id))
    }

    /// get_next_k_items(self, query_id, k, search_exp, max_increments, exclude_vec)
    ///
    /// Returns the next batch of `(score, item_id)` pairs. Same score
    /// convention as new_search. Also releases the GIL for the search.
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
        let results: Vec<(NotNan<f32>, u32)> = py.detach(|| {
            self.inner
                .get_next_k_items(query_id, k, search_exp, max_increments, &exclude_set)
        });

        Ok(results
            .into_iter()
            .map(|(nn, id)| (nn.into_inner(), id))
            .collect())
    }

    /// insert(self, embeddings: np.ndarray[f32, 2]) -> tuple[int, int]
    ///
    /// Assigns each embedding row the next available id, routes it to its
    /// nearest leaf, and appends it there. No rebalancing: a leaf that
    /// already has plenty of children just keeps growing. Safe to call
    /// concurrently with search or with another insert; two operations
    /// only serialize when they land on the same leaf. The caller is
    /// responsible for embeddings already matching this index's
    /// metric/is_normalized/dtype convention. Releases the GIL for the
    /// actual write.
    ///
    /// Returns `(start_id, end_id)`, the assigned ids as a half-open
    /// range (`end_id` excluded, so `range(start_id, end_id)` in Python
    /// gives every assigned id in row order). This is the caller's own
    /// mapping back to whatever external ids it uses.
    ///
    /// Not atomic: a crash partway through can leave next_item_id ahead
    /// of what actually landed on disk, permanently skipping the unwritten
    /// ids rather than reusing or colliding with one already written.
    /// total_items still counts only what was stored.
    fn insert(&self, py: Python<'_>, embeddings: PyReadonlyArray2<f32>) -> PyResult<(u32, u32)> {
        self.check_not_closed()?;
        let embeddings: Array2<f32> = embeddings.to_owned_array();
        let range = py.detach(|| self.inner.insert(embeddings));
        Ok((range.start, range.end))
    }

    /// cleanup_persisted_queries_older_than(self, cutoff_unix_secs: float) -> int
    ///
    /// Erases every persisted query older than cutoff_unix_secs (a Unix
    /// timestamp, e.g. datetime.datetime(...).timestamp()). Returns how
    /// many were erased. Releases the GIL for the actual scan.
    fn cleanup_persisted_queries_older_than(
        &self,
        py: Python<'_>,
        cutoff_unix_secs: f64,
    ) -> PyResult<usize> {
        self.check_not_closed()?;
        Ok(py.detach(|| {
            self.inner
                .cleanup_persisted_queries_older_than(cutoff_unix_secs as u64)
        }))
    }

    /// close(self)
    ///
    /// Persists every in-flight query to disk and marks this index closed;
    /// every other method raises ValueError afterward. Safe to call more
    /// than once. Releases the GIL for the actual persisting.
    fn close(&mut self, py: Python<'_>) {
        if self.closed {
            return;
        }
        py.detach(|| self.inner.shutdown());
        self.closed = true;
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
    ) {
        self.close(py);
    }
}
