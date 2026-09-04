
use zarrs::array::data_type::{float16, float32};
use zarrs::storage::ReadableListableStorage;
use zarrs::array::Array;
use ndarray::{Array2, Array1};

use half::f16;

pub struct Node {
    store: ReadableListableStorage,
    pub group_path: String,
    pub child_key: String,
    embeddings: Option<Array2<f32>>,
    children: Option<Array1<u32>>,
    checked_embs: bool,
    checked_childs: bool,
}

impl Node
{
    /// Creates a new Node instance.
    /// Returns:
    ///     Node<T>: A new instance of Node with the specified store, group path, and child key.
    pub fn new(store: ReadableListableStorage, group_path: String, child_key: String) -> Self {
        Node {
            store: store,
            group_path: group_path,
            child_key: child_key,
            embeddings: None,
            children: None,
            checked_embs: false,
            checked_childs: false,
            // _marker: PhantomData
        }
    }

    /// Retrieves the embeddings of the node.
    pub fn embeddings(&mut self) -> &Option<Array2<f32>> {
        if self.embeddings.is_none() && !self.checked_embs {
            let embeddings_path = format!("{}/embeddings", &self.group_path);
            // let arr = Array::open(self.store.clone(), &embeddings_path);
            let arr = Array::open(self.store.clone(), &embeddings_path);
            match arr {
                Ok(array) => {
                    let dtype = array.data_type();
                    if *dtype != float32() && *dtype != float16() {
                        panic!("unknown datatype")
                    }
                    self.embeddings = Some(
                        if *dtype == float32() {
                            array.retrieve_array_subset::<Array2<f32>>(&array.subset_all())
                                .expect("Failed to retrieve embeddings array")
                        } else {
                            array.retrieve_array_subset::<Array2<f16>>(&array.subset_all())
                                .expect("Failed to retrieve embeddings array")
                                .mapv(|x: f16| x.to_f32())
                        } 
                    )
                },
                Err(_) => self.embeddings = None,
            };
            self.checked_embs = true;
        }
        &self.embeddings
    }

    /// Retrieves the IDs of the children of the node.
    pub fn children(&mut self) -> &Option<Array1<u32>> {
        if self.children.is_none() && !self.checked_childs {
            let ids_path = format!("{}/{}", &self.group_path, &self.child_key);
            let arr = Array::open(self.store.clone(), &ids_path);
            match arr {
                Ok(array) => self.children = Some(
                        array.retrieve_array_subset::<Array1<u32>>(&array.subset_all())
                        .expect("Failed to retrieve ids array")
                    ),
                Err(_) => self.children = None,
            };
            self.checked_childs = true;
            // println!("children ({:?}): {:?}", self.group_path, &self.children.iter().len())
        }
        &self.children
    }

    /// Clears the cached embeddings and children of the node.
    /// This method is useful to free up memory if the node's data is no longer needed.
    /// A subsequent call to `embeddings()`/`children()` will re-fetch from the store.
    pub fn clear_cache(&mut self) {
        self.embeddings = None;
        self.children = None;
        self.checked_embs = false;
        self.checked_childs = false;
    }

    /// Checks if the node's embeddings or children are loaded.
    /// Returns:
    ///     bool: True if either embeddings and/or children are loaded, False otherwise.
    pub fn is_loaded(&self) -> bool {
        self.embeddings.is_some() || self.children.is_some()
    }

    /// Bytes currently held by this node's cached embeddings/children, for
    /// eviction-policy accounting.
    pub fn resident_bytes(&self) -> usize {
        let emb_bytes = self.embeddings.as_ref().map_or(0, |e| e.len() * size_of::<f32>());
        let child_bytes = self.children.as_ref().map_or(0, |c| c.len() * size_of::<u32>());
        emb_bytes + child_bytes
    }
}

#[cfg(test)]
#[path = "utests/node.rs"]
mod tests;
