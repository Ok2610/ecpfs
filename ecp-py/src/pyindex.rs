use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArrayMethods};
use ndarray::Array1;
use std::collections::HashSet;
use std::path::PathBuf;
use ordered_float::NotNan;
use ecp_core::search::Index;

#[pyclass(module = "ecp.index")]
pub struct IndexWrapper {
    inner: Index,
}

#[pymethods]
impl IndexWrapper {
    /// __new__(index_path: PathBuf, memory_limit_bytes: Optional[int] = None)
    ///
    /// Loads an index from disk, deriving its metric/levels/nodes from the
    /// store itself. memory_limit_bytes bounds how many touched nodes stay
    /// cached (LRU-evicted); None keeps every touched node cached forever.
    #[new]
    #[pyo3(signature = (index_path, memory_limit_bytes=None))]
    fn new(index_path: PathBuf, memory_limit_bytes: Option<usize>) -> PyResult<Self> {
        Ok(IndexWrapper { inner: Index::load(index_path, memory_limit_bytes) })
    }

    /// set_memory_limit_bytes(self, memory_limit_bytes: Optional[int])
    ///
    /// Raises or lowers the memory limit on an already-loaded index, no
    /// reload needed. Lowering below what's currently resident evicts
    /// immediately.
    fn set_memory_limit_bytes(&mut self, memory_limit_bytes: Option<usize>) {
        self.inner.set_memory_limit_bytes(memory_limit_bytes);
    }

    /// new_search(self, query: np.ndarray[f32, 1], k: int,
    ///            search_exp: u32, max_increments: i32)
    ///
    /// Returns `(items, query_id)`, where `items: List[(distance: float, item_id: int)]`.
    fn new_search(
        &mut self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude_vec: Vec<u32>,
    ) -> PyResult<(Vec<(f32, u32)>, usize)> {
        // to owned Array1<f32>
        let query: Array1<f32> = query.to_owned_array();
        let exclude_set: HashSet<u32> = exclude_vec.into_iter().collect();

        // call through to your Rust Index
        let (results, query_id): (Vec<(NotNan<f32>, u32)>, usize) =
            self.inner.new_search(query, k, search_exp, max_increments, &exclude_set);

        // unpack NotNan<f32> into raw f32
        let items: Vec<(f32, u32)> = results
            .into_iter()
            .map(|(nn, id)| (nn.into_inner(), id))
            .collect();

        Ok((items, query_id))
    }

    /// incremental_search(self, query_id, k, search_exp, max_increments)
    ///
    /// Returns the next batch of `(distance, item_id)` pairs.
    fn incremental_search(
        &mut self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude_vec: Vec<u32>,
    ) -> PyResult<Vec<(f32, u32)>> {
        let exclude_set: HashSet<u32> = exclude_vec.into_iter().collect();
        let results: Vec<(NotNan<f32>, u32)> =
            self.inner.get_next_k_items(query_id, k, search_exp, max_increments, &exclude_set);

        Ok(results
            .into_iter()
            .map(|(nn, id)| (nn.into_inner(), id))
            .collect())
    }
}