use std::sync::OnceLock;

use ndarray::{Array1, Array2};
use zarrs::array::{Array, ArrayCreateError};
use zarrs::storage::ReadableListableStorage;

use crate::dtype::read_subset_as_f32;
use crate::error::{EcpError, Result, ResultExt};

/// One node's embeddings and child ids, read from the store on first use and
/// then kept in memory.
pub struct Node {
    store: ReadableListableStorage,
    /// The node's group in the store, such as `/lvl_2/node_7`.
    pub group_path: String,
    /// `node_ids` for an internal node, `item_ids` for a leaf.
    pub child_key: String,
    embeddings: OnceLock<Result<Option<Array2<f32>>>>,
    children: OnceLock<Result<Option<Array1<u32>>>>,
}

/// Opens the array at path. Ok(None) if there is no array there,
/// Err if the store failed
fn open_if_present(
    store: &ReadableListableStorage,
    path: &str,
) -> Result<Option<Array<dyn zarrs::storage::ReadableListableStorageTraits>>> {
    match Array::open(store.clone(), path) {
        Ok(array) => Ok(Some(array)),
        Err(ArrayCreateError::MissingMetadata) => Ok(None),
        Err(e) => Err(EcpError::Store(format!("failed to open {path}: {e}"))),
    }
}

impl Node {
    /// Creates a `Node` for the group at `group_path`. Nothing is read until
    /// `embeddings` or `children` is called.
    pub fn new(store: ReadableListableStorage, group_path: String, child_key: String) -> Self {
        Node {
            store,
            group_path,
            child_key,
            embeddings: OnceLock::new(),
            children: OnceLock::new(),
        }
    }

    /// Returns the node's embeddings as f32, reading them on the first call.
    /// Ok(None) if the node was never written, Err if the read failed.
    /// Both results are cached.
    pub fn embeddings(&self) -> Result<Option<&Array2<f32>>> {
        let result = self.embeddings.get_or_init(|| {
            let embeddings_path = format!("{}/embeddings", self.group_path);
            match open_if_present(&self.store, &embeddings_path)? {
                Some(array) => Ok(Some(read_subset_as_f32(
                    &array,
                    &array.subset_all(),
                    &embeddings_path,
                )?)),
                None => Ok(None),
            }
        });
        match result {
            Ok(opt) => Ok(opt.as_ref()),
            Err(e) => Err(e.clone()),
        }
    }

    /// Returns the node's child ids as u32, reading them on the first call.
    /// `Ok(None)` means the node was never written, `Err` if the read failed.
    /// Both results are cached.
    pub fn children(&self) -> Result<Option<&Array1<u32>>> {
        let result = self.children.get_or_init(|| {
            let ids_path = format!("{}/{}", self.group_path, self.child_key);
            match open_if_present(&self.store, &ids_path)? {
                Some(array) => {
                    let ids = array
                        .retrieve_array_subset::<Array1<u32>>(&array.subset_all())
                        .store_err_with(|| format!("failed to retrieve {ids_path}"))?;
                    Ok(Some(ids))
                }
                None => Ok(None),
            }
        });
        match result {
            Ok(opt) => Ok(opt.as_ref()),
            Err(e) => Err(e.clone()),
        }
    }

    /// Checks whether `embeddings` or `children` has read data. Always false
    /// for a node missing on disk or one whose read failed.
    pub fn is_loaded(&self) -> bool {
        self.embeddings
            .get()
            .is_some_and(|r| matches!(r, Ok(Some(_))))
            || self
                .children
                .get()
                .is_some_and(|r| matches!(r, Ok(Some(_))))
    }

    /// Returns the bytes of embeddings and child ids this node holds in memory,
    /// which the node cache counts against its limit.
    pub fn resident_bytes(&self) -> usize {
        let emb_bytes = self
            .embeddings
            .get()
            .and_then(|r| r.as_ref().ok())
            .and_then(Option::as_ref)
            .map_or(0, |e| e.len() * size_of::<f32>());
        let child_bytes = self
            .children
            .get()
            .and_then(|r| r.as_ref().ok())
            .and_then(Option::as_ref)
            .map_or(0, |c| c.len() * size_of::<u32>());
        emb_bytes + child_bytes
    }
}

#[cfg(test)]
#[path = "utests/node.rs"]
mod tests;
