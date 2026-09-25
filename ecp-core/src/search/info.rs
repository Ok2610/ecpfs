use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use zarrs::array::{Array, ArrayCreateError};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorage;

use crate::error::{EcpError, Result, ResultExt};
use crate::format::{FORMAT_VERSION, V1_ARRAYS};
use crate::metric::Metric;

/// Opens the array at `path` in `store`, or returns `None` if there is no
/// such array.
fn open_optional(
    store: &ReadableListableStorage,
    path: &str,
) -> Result<Option<Array<dyn zarrs::storage::ReadableListableStorageTraits>>> {
    match Array::open(store.clone(), path) {
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
    open_optional(store, &format!("/info/{name}"))?
        .ok_or_else(|| EcpError::Corrupt(format!("not a valid index: /info/{name} is missing")))
}

/// Reads `info/format_version` and refuses a version this build does not read.
/// `None` means the field is missing from an index that has every other array
/// a version 1 index holds. If others are missing too, the error names them all.
pub(super) fn read_format_version(store: &ReadableListableStorage) -> Result<Option<u32>> {
    let Some(array) = open_optional(store, "/info/format_version")? else {
        let mut missing = vec!["/info/format_version"];
        for path in V1_ARRAYS {
            if open_optional(store, path)?.is_none() {
                missing.push(path);
            }
        }
        if missing.len() > 1 {
            return Err(EcpError::Corrupt(format!(
                "not a valid index, missing: {}",
                missing.join(", ")
            )));
        }
        return Ok(None);
    };

    let version = array
        .retrieve_array_subset::<Vec<u32>>(&array.subset_all())
        .store_err("failed to retrieve info/format_version")?[0];
    if version != FORMAT_VERSION {
        return Err(EcpError::UnsupportedVersion(format!(
            "index format version {version} is not supported, this build reads version {FORMAT_VERSION}"
        )));
    }
    Ok(Some(version))
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

    let is_normalized = match open_optional(store, "/info/is_normalized")? {
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
    /// The version of the on-disk format. An index missing the field but
    /// otherwise complete reads as the current version.
    pub format_version: u32,
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
        let format_version = read_format_version(&store)?.unwrap_or(FORMAT_VERSION);
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
            format_version,
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
