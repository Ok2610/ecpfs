use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use zarrs::array::{Array, ArrayCreateError};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorage;

use crate::error::{EcpError, Result, ResultExt};
use crate::metric::Metric;

/// Opens the array at `info/{name}` in `store`, or returns `None` if there is
/// no such array.
fn open_optional_info_field(
    store: &ReadableListableStorage,
    name: &str,
) -> Result<Option<Array<dyn zarrs::storage::ReadableListableStorageTraits>>> {
    let path = format!("/info/{name}");
    match Array::open(store.clone(), &path) {
        Ok(array) => Ok(Some(array)),
        Err(ArrayCreateError::MissingMetadata) => Ok(None),
        Err(other) => Err(EcpError::Store(format!("failed to open {path}: {other}"))),
    }
}

/// Opens the array at `info/{name}` in `store`. A missing array means the
/// store isn't a valid index, so this maps to [`EcpError::Corrupt`] rather
/// than [`EcpError::NotFound`].
fn open_info_field(
    store: &ReadableListableStorage,
    name: &str,
) -> Result<Array<dyn zarrs::storage::ReadableListableStorageTraits>> {
    open_optional_info_field(store, name)?
        .ok_or_else(|| EcpError::Corrupt(format!("not a valid index: /info/{name} is missing")))
}

/// Reads `info/levels`, `info/metric` and `info/is_normalized`, which is
/// `None` when the index does not record it.
pub(super) fn read_info_fields(
    store: &ReadableListableStorage,
) -> Result<(u32, Metric, Option<bool>)> {
    let levels_array = open_info_field(store, "levels")?;
    let levels: u32 = levels_array
        .retrieve_array_subset::<Vec<u32>>(&levels_array.subset_all())
        .store_err("failed to retrieve info/levels")?[0];

    let metric_array = open_info_field(store, "metric")?;
    let metric_str = metric_array
        .retrieve_array_subset::<Vec<String>>(&metric_array.subset_all())
        .store_err("failed to retrieve info/metric")?
        .remove(0);
    let metric = Metric::from_str(&metric_str)
        .map_err(|e| EcpError::Corrupt(format!("info/metric holds an unrecognized metric: {e}")))?;

    let is_normalized = match open_optional_info_field(store, "is_normalized")? {
        Some(array) => Some(
            array
                .retrieve_array_subset::<Vec<bool>>(&array.subset_all())
                .store_err("failed to retrieve info/is_normalized")?[0],
        ),
        None => None,
    };

    Ok((levels, metric, is_normalized))
}

/// Reads the `u32` scalar at `info/{name}`.
pub(super) fn read_info_u32(store: &ReadableListableStorage, name: &str) -> Result<u32> {
    let array = open_info_field(store, name)?;
    Ok(array
        .retrieve_array_subset::<Vec<u32>>(&array.subset_all())
        .store_err_with(|| format!("failed to retrieve info/{name}"))?[0])
}

/// An index's `info/*` fields and representative count, read without
/// loading the tree.
pub struct IndexInfo {
    /// Node levels below the root; the last one holds the leaves.
    pub levels: u32,
    pub metric: Metric,
    /// Whether every stored embedding is unit-length, or `None` if the index
    /// does not say.
    pub is_normalized: Option<bool>,
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
    pub fn load(index_path: PathBuf) -> Result<Self> {
        if !index_path.exists() {
            return Err(EcpError::NotFound(format!(
                "index directory {} does not exist",
                index_path.display()
            )));
        }
        let store: ReadableListableStorage =
            Arc::new(FilesystemStore::new(&index_path).store_err("failed to open store")?);
        Self::load_from_store(store)
    }

    /// Reads the info fields from `store` instead of a path. Otherwise the same as `load`.
    /// Note: This function is only split out from `load` for unit tests.
    fn load_from_store(store: ReadableListableStorage) -> Result<Self> {
        let (levels, metric, is_normalized) = read_info_fields(&store)?;
        let total_items = read_info_u32(&store, "total_items")?;
        let next_item_id = read_info_u32(&store, "next_item_id")?;

        let rep_ids_array = Array::open(store.clone(), "/rep_item_ids").map_err(|e| match e {
            ArrayCreateError::MissingMetadata => {
                EcpError::Corrupt("not a valid index: /rep_item_ids is missing".to_string())
            }
            other => EcpError::Store(format!("failed to open rep_item_ids: {other}")),
        })?;
        let total_representatives = rep_ids_array.shape()[0] as u32;

        Ok(IndexInfo {
            levels,
            metric,
            is_normalized,
            total_items,
            next_item_id,
            total_representatives,
        })
    }
}

#[cfg(test)]
#[path = "utests/info.rs"]
mod tests;
