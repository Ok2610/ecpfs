use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorage;

use crate::metric::Metric;

/// Reads `info/levels`, `info/metric` and `info/is_normalized`.
pub(super) fn read_info_fields(store: &ReadableListableStorage) -> (u32, Metric, bool) {
    let levels_array =
        Array::open(store.clone(), "/info/levels").expect("Failed to open info/levels");
    let levels: u32 = levels_array
        .retrieve_array_subset::<Vec<u32>>(&levels_array.subset_all())
        .expect("Failed to retrieve info/levels")[0];

    let metric_array =
        Array::open(store.clone(), "/info/metric").expect("Failed to open info/metric");
    let metric_str = metric_array
        .retrieve_array_subset::<Vec<String>>(&metric_array.subset_all())
        .expect("Failed to retrieve info/metric")
        .remove(0);
    let metric = Metric::from_str(&metric_str)
        .unwrap_or_else(|e| panic!("info/metric holds an unrecognized metric: {e}"));

    let is_normalized_array = Array::open(store.clone(), "/info/is_normalized")
        .expect("Failed to open info/is_normalized");
    let is_normalized: bool = is_normalized_array
        .retrieve_array_subset::<Vec<bool>>(&is_normalized_array.subset_all())
        .expect("Failed to retrieve info/is_normalized")[0];

    (levels, metric, is_normalized)
}

/// Reads the `u32` scalar at `info/{name}`.
pub(super) fn read_info_u32(store: &ReadableListableStorage, name: &str) -> u32 {
    let path = format!("/info/{name}");
    let array =
        Array::open(store.clone(), &path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));
    array
        .retrieve_array_subset::<Vec<u32>>(&array.subset_all())
        .unwrap_or_else(|e| panic!("Failed to retrieve {path}: {e}"))[0]
}

/// An index's `info/*` fields and representative count, read without
/// loading the tree.
pub struct IndexInfo {
    /// Node levels below the root; the last one holds the leaves.
    pub levels: u32,
    pub metric: Metric,
    /// Whether every stored embedding is unit-length.
    pub is_normalized: bool,
    /// Items stored in the index.
    pub total_items: u32,
    /// The id `insert` gives the next item. Equal to `total_items` unless a
    /// crash stopped an insert between taking ids and writing the items.
    pub next_item_id: u32,
    /// How many representatives the build picked, one per leaf node. Read
    /// from `/rep_item_ids`'s length.
    pub total_representatives: u32,
}

impl IndexInfo {
    /// Loads an index's info fields from `index_path`.
    pub fn load(index_path: PathBuf) -> Self {
        let store: ReadableListableStorage =
            Arc::new(FilesystemStore::new(&index_path).expect("Failed to open store"));
        Self::load_from_store(store)
    }

    /// Reads the info fields from `store` instead of a path. Otherwise the same as `load`.
    /// Note: This function is only split out from `load` for unit tests.
    fn load_from_store(store: ReadableListableStorage) -> Self {
        let (levels, metric, is_normalized) = read_info_fields(&store);
        let total_items = read_info_u32(&store, "total_items");
        let next_item_id = read_info_u32(&store, "next_item_id");

        let rep_ids_array =
            Array::open(store.clone(), "/rep_item_ids").expect("Failed to open rep_item_ids");
        let total_representatives = rep_ids_array.shape()[0] as u32;

        IndexInfo {
            levels,
            metric,
            is_normalized,
            total_items,
            next_item_id,
            total_representatives,
        }
    }
}

#[cfg(test)]
#[path = "utests/info.rs"]
mod tests;
