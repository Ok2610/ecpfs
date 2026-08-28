
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{as_readable_listable, new_memory_store, write_node, write_node_f16, write_node_unsupported_dtype};
    use ndarray::{array, Array1, Array2};

    #[test]
    fn loads_and_caches_embeddings_and_children() {
        let store = new_memory_store();
        let embeddings: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let children: Array1<u32> = array![10, 20];
        write_node(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

        let mut node = Node::new(
            as_readable_listable(&store),
            "/lvl_1/node_0".to_string(),
            "node_ids".to_string(),
        );

        assert!(!node.is_loaded());

        assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
        assert!(node.is_loaded());
        assert_eq!(node.children().as_ref().unwrap(), &children);

        node.clear_cache();
        assert!(!node.is_loaded());
        // Lazily reloads from the store after a cache clear.
        assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
    }

    /// Values chosen to be exactly representable in f16 (10 mantissa bits), so the
    /// upcast to f32 via `mapv(|x: f16| x.to_f32())` is exact, not approximate.
    #[test]
    fn loads_f16_embeddings_upcast_to_f32() {
        let store = new_memory_store();
        let embeddings: Array2<f32> = array![[1.0, 2.0], [3.5, -1.25]];
        let children: Array1<u32> = array![10, 20];
        write_node_f16(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

        let mut node = Node::new(
            as_readable_listable(&store),
            "/lvl_1/node_0".to_string(),
            "node_ids".to_string(),
        );

        assert_eq!(node.embeddings().as_ref().unwrap(), &embeddings);
    }

    /// f16 and f32 are the only supported embedding dtypes; anything else (e.g. an
    /// embeddings array left as float64 by an upstream caller that never cast it)
    /// must fail loudly at load time rather than silently truncating precision.
    #[test]
    #[should_panic(expected = "unknown datatype")]
    fn unsupported_dtype_panics_instead_of_silently_truncating() {
        let store = new_memory_store();
        let embeddings: Array2<f32> = array![[1.0, 2.0], [3.5, -1.25]];
        let children: Array1<u32> = array![10, 20];
        write_node_unsupported_dtype(&store, "/lvl_1/node_0", &embeddings, "node_ids", &children);

        let mut node = Node::new(
            as_readable_listable(&store),
            "/lvl_1/node_0".to_string(),
            "node_ids".to_string(),
        );

        node.embeddings();
    }

    #[test]
    fn missing_node_yields_none_without_panicking() {
        let store = new_memory_store();
        let mut node = Node::new(
            as_readable_listable(&store),
            "/lvl_1/node_absent".to_string(),
            "item_ids".to_string(),
        );

        assert!(node.embeddings().is_none());
        assert!(node.children().is_none());
        assert!(!node.is_loaded());
    }
}
