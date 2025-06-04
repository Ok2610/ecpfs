// use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use ndarray::Array1;

use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorage;

use std::collections::BinaryHeap;
use ordered_float::NotNan;

use crate::ecp_node::Node;
use crate::utils::HeapEntry;
use crate::utils::{calculate_distances, Metric};

struct QueryState {
    query: Array1<f32>,
    tree_pq: BinaryHeap<HeapEntry>,
    items: Vec<(NotNan<f32>, u32)>,
}

pub struct Index {
    metric: Metric,
    levels: u32,
    root: Array2<f32>,
    nodes: Vec<Vec<Node>>,
    queries: Vec<QueryState>,
    // queries: Vec<Array1<f32>>,
    // tree_pq: Vec<BinaryHeap<HeapEntry>>,
    // items: Vec<Vec<(NotNan<f32>, u32)>>,
}

impl Index 
{
    /// Creates a new Index instance.
    /// Returns:
    ///     Index<T>: A new instance of Index with the specified metric, levels, root, and nodes.
    pub fn new(
        index_path: PathBuf,
        metric: Metric,
        levels: u32,
        root: Array2<f32>,
        node_paths: Vec<Vec<String>>, 
    ) -> Self {
        let mut nodes = Vec::new();
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(&index_path).expect("Failed to open store"));
        for i in 0..levels {
            nodes.push(Vec::new());
            let mut c_key = "node_ids".to_string(); 
            if i+1 == levels {
                c_key = "item_ids".to_string();
            }
            for node_path in &node_paths[i as usize] {
                let node = Node::new(
                    store.clone(),
                    node_path.clone(),
                    c_key.clone(),
                );
                nodes[i as usize].push(node);
            }
        }
        Index {
            metric: metric.into(),
            levels: levels,
            root: root,
            nodes: nodes,
            queries: Vec::new(),
            // tree_pq: Vec::new(),
            // items: Vec::new(),
        }
    }

    pub fn new_search(
        &mut self,
        query: Array1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        // exclude: &HashSet<u32>,
    ) -> (Vec<(NotNan<f32>, u32)>, usize) {
        self.queries.push(QueryState {
            query: query,
            tree_pq: BinaryHeap::new(),
            items: Vec::new() 
        });
        // self.tree_pq.push(BinaryHeap::new());
        // self.items.push(Vec::new());
        // self.queries.push(query);
        let query_id = self.queries.len()-1;
        self.incremental_search(query_id, k, search_exp, max_increments);
        (self.get_next_k_items(query_id, k, search_exp, max_increments), query_id)
    }

    pub fn incremental_search(
        &mut self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        // exclude: &HashSet<u32>,
    ) -> () {
        let QueryState{ 
            query,
            tree_pq,
            items
        }: &mut QueryState = &mut self.queries[query_id];

        // This method will perform an incremental search on the index.
        // It will use the provided query and return the updated priority queues.
        let sign = match self.metric {
            Metric::L2 => -1.0,
            Metric::IP => 1.0,
            Metric::Cos => -1.0,
        };
        let mut search_exp = search_exp;

        let mut leaf_cnt = 0;
        let mut items_cnt = 0;
        let mut increments = 0;

        // Add root to tree if empty (new search)
        if tree_pq.is_empty() {
            let root_distances: Array1<f32> = calculate_distances(
                &self.root,
                &query,
                &self.metric
            );
            for i in 0..root_distances.len() {
                tree_pq.push(
                    HeapEntry {
                        score: NotNan::new(sign * root_distances[i]).unwrap(), 
                        is_leaf: false as i32, 
                        level: 0,
                        node_id: i as u32
                    });
            }
        }

        // Search tree
        while !tree_pq.is_empty() {
            let HeapEntry {
                score: _,
                is_leaf,
                level, 
                node_id
            } = tree_pq.pop().unwrap();
            let lvl = level as usize;
            let node = node_id as usize;
            let embeddings_f32: &Array2<f32> = match self.nodes[lvl][node].embeddings() {
                Some(embs) => embs,
                None => continue,
            };

            // let (top, distances): (Vec<usize>, Array1<f32>) = calculate_distances(
            let distances: Array1<f32> = calculate_distances(
                embeddings_f32,
                &query,
                &self.metric,
            );
            if is_leaf == 1 {
                let children = self.nodes[lvl][node].children().as_ref().unwrap();
                for i in 0..distances.len() {
                    // -1.0 * sign * distance : min sort Vec
                    items.push((NotNan::new(-1.0 * sign * distances[i]).unwrap(), children[i]));
                    items_cnt += 1;
                }
                leaf_cnt += 1;
            } else {
                let children = self.nodes[lvl][node].children().as_ref().unwrap();
                for i in 0..distances.len() {
                    // sign * distance : max sort heap queue
                    if (level + 1) == (self.levels - 1) {
                        tree_pq.push(
                            HeapEntry {
                                score: NotNan::new(sign * distances[i]).unwrap(),
                                is_leaf: true as i32,
                                level: level+1,
                                node_id: children[i]
                            });
                    } else {
                        tree_pq.push(
                            HeapEntry { 
                                score: NotNan::new(sign * distances[i]).unwrap(),
                                is_leaf: false as i32, 
                                level: level + 1,
                                node_id: children[i],
                            });
                    }
                }
            }

            if leaf_cnt == search_exp {
                if items_cnt >= k {
                    items.sort_unstable_by_key(|&(first, _)| first);
                    // println!("Tree_PQ: {:?}, Items: {:?}", tree_pq.len(), items.len());
                    break
                }
                if increments > max_increments || max_increments == -1 {
                    increments += 1;
                    search_exp *= 2;
                } else {
                    break
                }
            }
        }
    }

    pub fn get_next_k_items(
        &mut self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32
    ) -> Vec<(NotNan<f32>, u32)> {
        let cnt = self.queries[query_id].items.len().min(k);
        if cnt == 0 && !self.queries[query_id].tree_pq.is_empty() {
            self.incremental_search(query_id, k, search_exp, max_increments);
        }
        self.queries[query_id].items.drain(0..cnt).collect()
    }
}