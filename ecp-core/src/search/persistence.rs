use std::collections::BinaryHeap;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::Array1;
use ordered_float::NotNan;
use zarrs::array::data_type::{float32, int32, uint32, uint64};
use zarrs::array::{Array, ArrayBuilder};
use zarrs::storage::{
    ReadableWritableListableStorage, StorePrefix, WritableStorageTraits, discover_children,
};

use super::QueryState;
use crate::utils::HeapEntry;

fn group_path(query_id: usize) -> String {
    format!("/queries/{query_id}")
}

/// Writes `/queries/{query_id}/*`, replacing anything already there for
/// this id.
pub(super) fn persist_query(
    store: &ReadableWritableListableStorage,
    query_id: usize,
    state: &QueryState,
) {
    erase_query(store, query_id);
    let base = group_path(query_id);

    write_f32_array(store, &format!("{base}/query"), state.query.to_vec());

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
    write_f32_array(store, &format!("{base}/tree_pq_score"), tree_pq_score);
    write_i32_array(store, &format!("{base}/tree_pq_is_leaf"), tree_pq_is_leaf);
    write_u32_array(store, &format!("{base}/tree_pq_level"), tree_pq_level);
    write_u32_array(store, &format!("{base}/tree_pq_node_id"), tree_pq_node_id);

    let mut items_score = Vec::with_capacity(state.items.len());
    let mut items_id = Vec::with_capacity(state.items.len());
    for (score, id) in &state.items {
        items_score.push(score.into_inner());
        items_id.push(*id);
    }
    write_f32_array(store, &format!("{base}/items_score"), items_score);
    write_u32_array(store, &format!("{base}/items_id"), items_id);

    write_persisted_at(store, &format!("{base}/persisted_at"), now_unix_secs());
}

/// Removes `/queries/{query_id}/*`, if present. A no-op if it doesn't
/// exist, `erase_prefix` on a nonexistent prefix is `Ok(())`.
pub(super) fn erase_query(store: &ReadableWritableListableStorage, query_id: usize) {
    let prefix = StorePrefix::new(format!("queries/{query_id}/"))
        .expect("a query_id-derived prefix is always valid");
    store
        .erase_prefix(&prefix)
        .expect("failed to erase query group");
}

/// `persist_query` if `state` has anything worth resuming, `erase_query`
/// otherwise.
pub(super) fn persist_or_erase(
    store: &ReadableWritableListableStorage,
    query_id: usize,
    state: &QueryState,
) {
    if state.tree_pq.is_empty() && state.items.is_empty() {
        erase_query(store, query_id);
    } else {
        persist_query(store, query_id, state);
    }
}

/// Every `query_id` with a persisted group under `/queries/` (empty if the
/// subtree doesn't exist). Listing only, no array reads.
pub(super) fn query_ids_on_disk(store: &ReadableWritableListableStorage) -> Vec<usize> {
    let prefix = StorePrefix::new("queries/").expect("\"queries/\" is a valid prefix");
    discover_children(store, &prefix)
        .expect("failed to list /queries")
        .iter()
        .filter_map(|child| {
            child
                .as_str()
                .trim_start_matches("queries/")
                .trim_end_matches('/')
                .parse::<usize>()
                .ok()
        })
        .collect()
}

/// Erases every persisted query whose `persisted_at` is strictly before
/// `cutoff_unix_secs`. Returns how many were erased.
pub(super) fn cleanup_older_than(
    store: &ReadableWritableListableStorage,
    cutoff_unix_secs: u64,
) -> usize {
    let mut erased = 0;
    for query_id in query_ids_on_disk(store) {
        let base = group_path(query_id);
        let persisted_at = read_persisted_at(store, &format!("{base}/persisted_at"));
        if persisted_at < cutoff_unix_secs {
            erase_query(store, query_id);
            erased += 1;
        }
    }
    erased
}

/// Loads one persisted query's full state, `None` if `/queries/{query_id}/`
/// doesn't exist.
pub(super) fn load_query(
    store: &ReadableWritableListableStorage,
    query_id: usize,
) -> Option<QueryState> {
    let base = group_path(query_id);
    let query_path = format!("{base}/query");
    if Array::open(store.clone(), &query_path).is_err() {
        return None;
    }

    let query = Array1::from_vec(read_f32_array(store, &query_path));

    let scores = read_f32_array(store, &format!("{base}/tree_pq_score"));
    let is_leaf = read_i32_array(store, &format!("{base}/tree_pq_is_leaf"));
    let level = read_u32_array(store, &format!("{base}/tree_pq_level"));
    let node_id = read_u32_array(store, &format!("{base}/tree_pq_node_id"));
    let tree_pq: BinaryHeap<HeapEntry> = scores
        .into_iter()
        .zip(is_leaf)
        .zip(level)
        .zip(node_id)
        .map(|(((score, is_leaf), level), node_id)| HeapEntry {
            score: NotNan::new(score).expect("a persisted score is never NaN"),
            is_leaf,
            level,
            node_id,
        })
        .collect();

    let items_score = read_f32_array(store, &format!("{base}/items_score"));
    let items_id = read_u32_array(store, &format!("{base}/items_id"));
    let items: Vec<(NotNan<f32>, u32)> = items_score
        .into_iter()
        .zip(items_id)
        .map(|(score, id)| {
            (
                NotNan::new(score).expect("a persisted score is never NaN"),
                id,
            )
        })
        .collect();

    Some(QueryState {
        query,
        tree_pq,
        items,
    })
}

fn write_f32_array(store: &ReadableWritableListableStorage, path: &str, data: Vec<f32>) {
    let len = (data.len() as u64).max(1);
    let array = ArrayBuilder::new(vec![data.len() as u64], vec![len], float32(), 0.0f32)
        .build(store.clone(), path)
        .expect("failed to build array");
    array
        .store_metadata()
        .expect("failed to store array metadata");
    array
        .store_array_subset(&array.subset_all(), Array1::from_vec(data))
        .expect("failed to store array");
}

fn write_i32_array(store: &ReadableWritableListableStorage, path: &str, data: Vec<i32>) {
    let len = (data.len() as u64).max(1);
    let array = ArrayBuilder::new(vec![data.len() as u64], vec![len], int32(), 0i32)
        .build(store.clone(), path)
        .expect("failed to build array");
    array
        .store_metadata()
        .expect("failed to store array metadata");
    array
        .store_array_subset(&array.subset_all(), Array1::from_vec(data))
        .expect("failed to store array");
}

fn write_u32_array(store: &ReadableWritableListableStorage, path: &str, data: Vec<u32>) {
    let len = (data.len() as u64).max(1);
    let array = ArrayBuilder::new(vec![data.len() as u64], vec![len], uint32(), 0u32)
        .build(store.clone(), path)
        .expect("failed to build array");
    array
        .store_metadata()
        .expect("failed to store array metadata");
    array
        .store_array_subset(&array.subset_all(), Array1::from_vec(data))
        .expect("failed to store array");
}

fn read_f32_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<f32> {
    let array = Array::open(store.clone(), path).expect("failed to open array");
    array
        .retrieve_array_subset::<Vec<f32>>(&array.subset_all())
        .expect("failed to retrieve array")
}

fn read_i32_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<i32> {
    let array = Array::open(store.clone(), path).expect("failed to open array");
    array
        .retrieve_array_subset::<Vec<i32>>(&array.subset_all())
        .expect("failed to retrieve array")
}

fn read_u32_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<u32> {
    let array = Array::open(store.clone(), path).expect("failed to open array");
    array
        .retrieve_array_subset::<Vec<u32>>(&array.subset_all())
        .expect("failed to retrieve array")
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after 1970")
        .as_secs()
}

fn write_persisted_at(store: &ReadableWritableListableStorage, path: &str, unix_secs: u64) {
    let shape: Vec<u64> = vec![];
    let array = ArrayBuilder::new(shape.clone(), shape, uint64(), 0u64)
        .build(store.clone(), path)
        .expect("failed to build array");
    array
        .store_metadata()
        .expect("failed to store array metadata");
    array
        .store_chunk(&[], vec![unix_secs])
        .expect("failed to store array");
}

fn read_persisted_at(store: &ReadableWritableListableStorage, path: &str) -> u64 {
    let array = Array::open(store.clone(), path).expect("failed to open array");
    array
        .retrieve_array_subset::<Vec<u64>>(&array.subset_all())
        .expect("failed to retrieve array")[0]
}

#[cfg(test)]
#[path = "utests/persistence.rs"]
mod tests;
