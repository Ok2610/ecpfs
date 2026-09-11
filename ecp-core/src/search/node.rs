use std::sync::OnceLock;

use ndarray::{Array1, Array2};
use zarrs::array::Array;
use zarrs::array::data_type::{float16, float32};
use zarrs::storage::ReadableListableStorage;

use half::f16;

/// One node's embeddings and children, lazily read from the store and
/// cached on first access.
pub struct Node {
    store: ReadableListableStorage,
    pub group_path: String,
    pub child_key: String,
    embeddings: OnceLock<Option<Array2<f32>>>,
    children: OnceLock<Option<Array1<u32>>>,
}

impl Node {
    pub fn new(store: ReadableListableStorage, group_path: String, child_key: String) -> Self {
        Node {
            store,
            group_path,
            child_key,
            embeddings: OnceLock::new(),
            children: OnceLock::new(),
        }
    }

    /// Lazily loads and upcasts `embeddings` to f32 on first call; `None`
    /// if the array doesn't exist. Cached after the first call either way,
    /// so a missing node isn't re-queried against the store.
    pub fn embeddings(&self) -> &Option<Array2<f32>> {
        self.embeddings.get_or_init(|| {
            let embeddings_path = format!("{}/embeddings", self.group_path);
            match Array::open(self.store.clone(), &embeddings_path) {
                Ok(array) => {
                    let dtype = array.data_type();
                    if *dtype != float32() && *dtype != float16() {
                        panic!("unsupported embeddings dtype: {dtype:?} (use float32 or float16)")
                    }
                    Some(if *dtype == float32() {
                        array
                            .retrieve_array_subset::<Array2<f32>>(&array.subset_all())
                            .expect("Failed to retrieve embeddings array")
                    } else {
                        array
                            .retrieve_array_subset::<Array2<f16>>(&array.subset_all())
                            .expect("Failed to retrieve embeddings array")
                            .mapv(|x: f16| x.to_f32())
                    })
                }
                Err(_) => None,
            }
        })
    }

    /// Lazily loads `child_key` on first call; `None` if the array doesn't
    /// exist. Cached after the first call either way, so a missing node
    /// isn't re-queried against the store.
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

    /// True if `embeddings`/`children` currently hold data; never true for
    /// a node confirmed missing from the store, even after it's been queried.
    pub fn is_loaded(&self) -> bool {
        self.embeddings.get().is_some_and(Option::is_some)
            || self.children.get().is_some_and(Option::is_some)
    }

    /// Bytes currently held by this node's cached embeddings/children, for
    /// eviction-policy accounting.
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
