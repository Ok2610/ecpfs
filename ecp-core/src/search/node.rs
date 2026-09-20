use std::sync::OnceLock;

use ndarray::{Array1, Array2};
use zarrs::array::Array;
use zarrs::storage::ReadableListableStorage;

use crate::dtype::read_subset_as_f32;

/// One node's embeddings and child ids, read from the store on first use and
/// then kept in memory.
pub struct Node {
    store: ReadableListableStorage,
    /// The node's group in the store, such as `/lvl_2/node_7`.
    pub group_path: String,
    /// `node_ids` for an internal node, `item_ids` for a leaf.
    pub child_key: String,
    embeddings: OnceLock<Option<Array2<f32>>>,
    children: OnceLock<Option<Array1<u32>>>,
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

    /// Returns the node's embeddings as f32, reading them on the first call, or
    /// `None` if the node has none on disk. Either result is kept, so a missing
    /// node is only looked up once.
    pub fn embeddings(&self) -> &Option<Array2<f32>> {
        self.embeddings.get_or_init(|| {
            let embeddings_path = format!("{}/embeddings", self.group_path);
            match Array::open(self.store.clone(), &embeddings_path) {
                Ok(array) => Some(read_subset_as_f32(
                    &array,
                    &array.subset_all(),
                    &embeddings_path,
                )),
                Err(_) => None,
            }
        })
    }

    /// Returns the node's child ids (its `child_key` array), reading them on the
    /// first call, or `None` if the node has none on disk. Either result is kept,
    /// so a missing node is only looked up once.
    pub fn children(&self) -> &Option<Array1<u32>> {
        self.children.get_or_init(|| {
            let ids_path = format!("{}/{}", self.group_path, self.child_key);
            match Array::open(self.store.clone(), &ids_path) {
                Ok(array) => Some(
                    array
                        .retrieve_array_subset::<Array1<u32>>(&array.subset_all())
                        .expect("Failed to retrieve ids array"),
                ),
                Err(_) => None,
            }
        })
    }

    /// Checks whether `embeddings` or `children` has read data. Always false
    /// for a node missing on disk.
    pub fn is_loaded(&self) -> bool {
        self.embeddings.get().is_some_and(Option::is_some)
            || self.children.get().is_some_and(Option::is_some)
    }

    /// Returns the bytes of embeddings and child ids this node holds in memory,
    /// which the node cache counts against its limit.
    pub fn resident_bytes(&self) -> usize {
        let emb_bytes = self
            .embeddings
            .get()
            .and_then(Option::as_ref)
            .map_or(0, |e| e.len() * size_of::<f32>());
        let child_bytes = self
            .children
            .get()
            .and_then(Option::as_ref)
            .map_or(0, |c| c.len() * size_of::<u32>());
        emb_bytes + child_bytes
    }
}

#[cfg(test)]
#[path = "utests/node.rs"]
mod tests;
