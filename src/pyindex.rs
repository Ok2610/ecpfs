use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use ndarray::{Array1, Array2};
use std::collections::HashSet;
use std::path::PathBuf;
use ordered_float::NotNan;
use crate::ecp_index::Index;
use crate::utils::Metric;

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s {
        "L2"  => Ok(Metric::L2),
        "IP"  => Ok(Metric::IP),
        "COS" => Ok(Metric::Cos),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown metric `{}` (use \"L2\", \"IP\", or \"Cos\")",
            other
        ))),
    }
}

#[pyclass(module = "ecp_index_rs.index")]
pub struct IndexWrapper {
    inner: Index,
}

#[pymethods]
impl IndexWrapper {
    /// __new__(index_path: PathBuf, metric: str, levels: int,
    ///        root: np.ndarray[f32, 2], node_paths: List[List[str]])
    #[new]
    fn new(
        index_path: PathBuf,
        metric: &str,
        levels: u32,
        root: PyReadonlyArray2<f32>,
        node_paths: Vec<Vec<String>>,
    ) -> PyResult<Self> {
        // parse your metric string into whatever enum/struct you use
        let metric = parse_metric(metric)?;
        // convert the numpy array into an owned `ndarray::Array2<f32>`
        let root: Array2<f32> = root.to_owned_array();

        // build your Index (now always f32)
        let inner = Index::new(index_path, metric, levels, root, node_paths);
        Ok(IndexWrapper { inner })
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