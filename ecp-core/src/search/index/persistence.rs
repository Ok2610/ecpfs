use std::collections::BinaryHeap;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::Array1;
use ordered_float::NotNan;
use zarrs::array::data_type::{float32, int32, uint32, uint64};
use zarrs::array::{Array, ArrayBuilder, ArrayCreateError};
use zarrs::storage::{
    ReadableWritableListableStorage, StorePrefix, WritableStorageTraits, discover_children,
};

use super::query::{HeapEntry, QueryState};
use crate::error::{EcpError, Result, ResultExt};

/// Returns the store path that query `query_id` is saved under.
fn group_path(query_id: usize) -> String {
    format!("/queries/{query_id}")
}

/// Saves `state` under `/queries/{query_id}/`, replacing any earlier save of
/// that query.
pub(super) fn persist_query(
    store: &ReadableWritableListableStorage,
    query_id: usize,
    state: &QueryState,
) -> Result<()> {
    erase_query(store, query_id)?;
    let base = group_path(query_id);

    write_f32_array(store, &format!("{base}/query"), state.query.to_vec())?;

    // One array per HeapEntry field, and per items tuple field
    let mut tree_pq_score = Vec::with_capacity(state.tree_pq.len());
    let mut tree_pq_is_leaf = Vec::with_capacity(state.tree_pq.len());
    let mut tree_pq_level = Vec::with_capacity(state.tree_pq.len());
    let mut tree_pq_node_id = Vec::with_capacity(state.tree_pq.len());
    for entry in &state.tree_pq {
        tree_pq_score.push(entry.score.into_inner());
        tree_pq_is_leaf.push(entry.is_leaf);
        tree_pq_level.push(entry.level);
        tree_pq_node_id.push(entry.node_id);
    }
    write_f32_array(store, &format!("{base}/tree_pq_score"), tree_pq_score)?;
    write_i32_array(store, &format!("{base}/tree_pq_is_leaf"), tree_pq_is_leaf)?;
    write_u32_array(store, &format!("{base}/tree_pq_level"), tree_pq_level)?;
    write_u32_array(store, &format!("{base}/tree_pq_node_id"), tree_pq_node_id)?;

    let mut items_score = Vec::with_capacity(state.items.len());
    let mut items_id = Vec::with_capacity(state.items.len());
    for (score, id) in &state.items {
        items_score.push(score.into_inner());
        items_id.push(*id);
    }
    write_f32_array(store, &format!("{base}/items_score"), items_score)?;
    write_u32_array(store, &format!("{base}/items_id"), items_id)?;

    write_persisted_at(store, &format!("{base}/persisted_at"), now_unix_secs())
}

/// Removes `/queries/{query_id}/*`; a no-op if it doesn't exist.
pub(super) fn erase_query(store: &ReadableWritableListableStorage, query_id: usize) -> Result<()> {
    let prefix = StorePrefix::new(format!("queries/{query_id}/"))
        .expect("a query_id-derived prefix is always valid");
    store
        .erase_prefix(&prefix)
        .store_err("failed to erase query group")
}

/// Saves `state`, or erases the saved copy when `state` has nothing left to
/// explore or return.
pub(super) fn persist_or_erase(
    store: &ReadableWritableListableStorage,
    query_id: usize,
    state: &QueryState,
) -> Result<()> {
    if state.tree_pq.is_empty() && state.items.is_empty() {
        erase_query(store, query_id)
    } else {
        persist_query(store, query_id, state)
    }
}

/// Lists the id of every query saved under `/queries/`, without reading any
/// arrays.
pub(super) fn query_ids_on_disk(store: &ReadableWritableListableStorage) -> Result<Vec<usize>> {
    let prefix = StorePrefix::new("queries/").expect("\"queries/\" is a valid prefix");
    Ok(discover_children(store, &prefix)
        .store_err("failed to list /queries")?
        .iter()
        .filter_map(|child| {
            child
                .as_str()
                .trim_start_matches("queries/")
                .trim_end_matches('/')
                .parse::<usize>()
                .ok()
        })
        .collect())
}

/// Erases every saved query whose save time (`persisted_at`) is before
/// `cutoff_unix_secs`. Returns how many were erased.
pub(super) fn cleanup_older_than(
    store: &ReadableWritableListableStorage,
    cutoff_unix_secs: u64,
) -> Result<usize> {
    let mut erased = 0;
    for query_id in query_ids_on_disk(store)? {
        let base = group_path(query_id);
        let persisted_at = read_persisted_at(store, &format!("{base}/persisted_at"))?;
        if persisted_at < cutoff_unix_secs {
            erase_query(store, query_id)?;
            erased += 1;
        }
    }
    Ok(erased)
}

/// Reads a saved query's state back. `Ok(None)` if that query was never
/// saved, `Err` if the store or a saved value can't be read.
pub(super) fn load_query(
    store: &ReadableWritableListableStorage,
    query_id: usize,
) -> Result<Option<QueryState>> {
    let base = group_path(query_id);
    let query_path = format!("{base}/query");
    match Array::open(store.clone(), &query_path) {
        Ok(_) => {}
        Err(ArrayCreateError::MissingMetadata) => return Ok(None),
        Err(e) => return Err(EcpError::Store(format!("failed to open {query_path}: {e}"))),
    }

    let query = Array1::from_vec(read_f32_array(store, &query_path)?);

    let scores = read_f32_array(store, &format!("{base}/tree_pq_score"))?;
    let is_leaf = read_i32_array(store, &format!("{base}/tree_pq_is_leaf"))?;
    let level = read_u32_array(store, &format!("{base}/tree_pq_level"))?;
    let node_id = read_u32_array(store, &format!("{base}/tree_pq_node_id"))?;
    let tree_pq: BinaryHeap<HeapEntry> = scores
        .into_iter()
        .zip(is_leaf)
        .zip(level)
        .zip(node_id)
        .map(|(((score, is_leaf), level), node_id)| {
            Ok(HeapEntry {
                score: NotNan::new(score)
                    .map_err(|_| EcpError::Corrupt(format!("{base}: saved score is NaN")))?,
                is_leaf,
                level,
                node_id,
            })
        })
        .collect::<Result<_>>()?;

    let items_score = read_f32_array(store, &format!("{base}/items_score"))?;
    let items_id = read_u32_array(store, &format!("{base}/items_id"))?;
    let items: Vec<(NotNan<f32>, u32)> = items_score
        .into_iter()
        .zip(items_id)
        .map(|(score, id)| {
            let score = NotNan::new(score)
                .map_err(|_| EcpError::Corrupt(format!("{base}: saved score is NaN")))?;
            Ok((score, id))
        })
        .collect::<Result<_>>()?;

    Ok(Some(QueryState {
        query,
        tree_pq,
        items,
    }))
}

/// Writes `data` as a one-chunk f32 array at `path`.
fn write_f32_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    data: Vec<f32>,
) -> Result<()> {
    let len = (data.len() as u64).max(1);
    let array = ArrayBuilder::new(vec![data.len() as u64], vec![len], float32(), 0.0f32)
        .build(store.clone(), path)
        .store_err("failed to build array")?;
    array
        .store_metadata()
        .store_err("failed to store array metadata")?;
    array
        .store_array_subset(&array.subset_all(), Array1::from_vec(data))
        .store_err("failed to store array")
}

/// Writes `data` as a one-chunk i32 array at `path`.
fn write_i32_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    data: Vec<i32>,
) -> Result<()> {
    let len = (data.len() as u64).max(1);
    let array = ArrayBuilder::new(vec![data.len() as u64], vec![len], int32(), 0i32)
        .build(store.clone(), path)
        .store_err("failed to build array")?;
    array
        .store_metadata()
        .store_err("failed to store array metadata")?;
    array
        .store_array_subset(&array.subset_all(), Array1::from_vec(data))
        .store_err("failed to store array")
}

/// Writes `data` as a one-chunk u32 array at `path`.
fn write_u32_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    data: Vec<u32>,
) -> Result<()> {
    let len = (data.len() as u64).max(1);
    let array = ArrayBuilder::new(vec![data.len() as u64], vec![len], uint32(), 0u32)
        .build(store.clone(), path)
        .store_err("failed to build array")?;
    array
        .store_metadata()
        .store_err("failed to store array metadata")?;
    array
        .store_array_subset(&array.subset_all(), Array1::from_vec(data))
        .store_err("failed to store array")
}

/// Reads the whole f32 array at `path`.
fn read_f32_array(store: &ReadableWritableListableStorage, path: &str) -> Result<Vec<f32>> {
    let array = Array::open(store.clone(), path).store_err("failed to open array")?;
    array
        .retrieve_array_subset::<Vec<f32>>(&array.subset_all())
        .store_err("failed to retrieve array")
}

/// Reads the whole i32 array at `path`.
fn read_i32_array(store: &ReadableWritableListableStorage, path: &str) -> Result<Vec<i32>> {
    let array = Array::open(store.clone(), path).store_err("failed to open array")?;
    array
        .retrieve_array_subset::<Vec<i32>>(&array.subset_all())
        .store_err("failed to retrieve array")
}

/// Reads the whole u32 array at `path`.
fn read_u32_array(store: &ReadableWritableListableStorage, path: &str) -> Result<Vec<u32>> {
    let array = Array::open(store.clone(), path).store_err("failed to open array")?;
    array
        .retrieve_array_subset::<Vec<u32>>(&array.subset_all())
        .store_err("failed to retrieve array")
}

/// Returns the current time in seconds since the Unix epoch.
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after 1970")
        .as_secs()
}

/// Records a query's save time, `unix_secs`, as a scalar at `path`.
fn write_persisted_at(
    store: &ReadableWritableListableStorage,
    path: &str,
    unix_secs: u64,
) -> Result<()> {
    let shape: Vec<u64> = vec![];
    let array = ArrayBuilder::new(shape.clone(), shape, uint64(), 0u64)
        .build(store.clone(), path)
        .store_err("failed to build array")?;
    array
        .store_metadata()
        .store_err("failed to store array metadata")?;
    array
        .store_chunk(&[], vec![unix_secs])
        .store_err("failed to store array")
}

/// Reads back a save time written by `write_persisted_at`.
fn read_persisted_at(store: &ReadableWritableListableStorage, path: &str) -> Result<u64> {
    let array = Array::open(store.clone(), path).store_err("failed to open array")?;
    Ok(array
        .retrieve_array_subset::<Vec<u64>>(&array.subset_all())
        .store_err("failed to retrieve array")?[0])
}

#[cfg(test)]
#[path = "utests/persistence.rs"]
mod tests;
