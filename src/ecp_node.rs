
use zarrs::array::DataType;
use zarrs::storage::ReadableListableStorage;
use zarrs::array::Array;
// use zarrs::array::ElementOwned;
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
                    self.embeddings = Some(
                        match array.data_type() {
                            DataType::Float32 => {
                                array.retrieve_array_subset_ndarray(&array.subset_all())
                                .expect("Failed to retrieve embeddings array")
                                .into_shape_clone(
                                    (array.shape()[0] as usize, 
                                    array.shape()[1] as usize)
                                )
                                .expect("Failed to reshape embeddings array")
                            },
                            DataType::Float16 => {
                                array.retrieve_array_subset_ndarray(&array.subset_all())
                                .expect("Failed to retrieve embeddings array")
                                .into_shape_clone(
                                    (array.shape()[0] as usize, 
                                    array.shape()[1] as usize)
                                )
                                .expect("Failed to reshape embeddings array")
                                .mapv(|x:f16| x.to_f32())
                            },
                            DataType::Float64 => {
                                array.retrieve_array_subset_ndarray(&array.subset_all())
                                .expect("Failed to retrieve embeddings array")
                                .into_shape_clone(
                                    (array.shape()[0] as usize, 
                                    array.shape()[1] as usize)
                                )
                                .expect("Failed to reshape embeddings array")
                                .mapv(|x:f64| x as f32)
                            },
                            _ => panic!("unknown datatype")
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
                        array.retrieve_array_subset_ndarray(&array.subset_all())
                        .expect("Failed to retrieve ids array")
                        .into_shape_clone(array.shape()[0] as usize)
                        .expect("Failed to reshape ids array")
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
    pub fn clear_cache(&mut self) {
        self.embeddings = None;
        self.children = None;
    }

    /// Checks if the node's embeddings or children are loaded.
    /// Returns:
    ///     bool: True if either embeddings and/or children are loaded, False otherwise.
    pub fn is_loaded(&self) -> bool {
        self.embeddings.is_some() || self.children.is_some()
    }
}