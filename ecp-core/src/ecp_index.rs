use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use ndarray::Array2;
use ndarray::Array1;

use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ListableStorageTraits, ReadableListableStorage, StorePrefix};

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
}

impl Index
{
    /// Loads an index from `index_path`, deriving `metric`, `levels`,
    /// `root`, and every level's node paths from the store itself
    /// (`info/levels`, `info/metric`, `index_root/embeddings`, and each
    /// `lvl_N/node_M` group).
    pub fn load(index_path: PathBuf) -> Self {
        let store: ReadableListableStorage =
            Arc::new(FilesystemStore::new(&index_path).expect("Failed to open store"));
        Self::load_from_store(store)
    }

    fn load_from_store(store: ReadableListableStorage) -> Self {
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

        let root_array = Array::open(store.clone(), "/index_root/embeddings")
            .expect("Failed to open index_root/embeddings");
        let root: Array2<f32> = root_array
            .retrieve_array_subset::<Array2<f32>>(&root_array.subset_all())
            .expect("Failed to retrieve index_root/embeddings");

        let mut nodes = Vec::with_capacity(levels as usize);
        for l in 0..levels {
            let lvl_name = format!("lvl_{}", l + 1);
            let prefix = StorePrefix::new(format!("{lvl_name}/"))
                .expect("level name produces an invalid store prefix");
            let listing = store
                .list_dir(&prefix)
                .unwrap_or_else(|e| panic!("Failed to list {lvl_name}: {e}"));

            let mut level_nodes: Vec<(u32, String)> = listing
                .prefixes()
                .iter()
                .filter_map(|p| {
                    let name = p.as_str().trim_end_matches('/').rsplit('/').next()?;
                    let idx: u32 = name.strip_prefix("node_")?.parse().ok()?;
                    Some((idx, format!("/{lvl_name}/{name}")))
                })
                .collect();
            level_nodes.sort_unstable_by_key(|(idx, _)| *idx);

            let c_key = if l + 1 == levels { "item_ids" } else { "node_ids" };
            nodes.push(
                level_nodes
                    .into_iter()
                    .map(|(_, path)| Node::new(store.clone(), path, c_key.to_string()))
                    .collect(),
            );
        }

        Index { metric, levels, root, nodes, queries: Vec::new() }
    }

    pub fn new_search(
        &mut self,
        query: Array1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
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
        self.incremental_search(query_id, k, search_exp, max_increments, exclude);
        (self.get_next_k_items(query_id, k, search_exp, max_increments, exclude), query_id)
    }

    pub fn incremental_search(
        &mut self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
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
            Metric::Cos => 1.0,
        };
        let mut search_exp = search_exp;

        let mut leaf_cnt = 0;
        let mut increments = 0;

        // Add root to tree if empty (new search)
        if tree_pq.is_empty() {
            let root_distances: Array1<f32> = calculate_distances(
                &self.root,
                &query,
                &self.metric
            );
            // A 1-level index is IVF-style: node_size == total_clusters, so root
            // already holds every leader and `nodes[0]` is the only (leaf) level.
            // Root entries must be marked as leaves from the start in that case,
            // since there is no intermediate level left to descend through.
            let is_root_leaf = self.levels == 1;
            for i in 0..root_distances.len() {
                tree_pq.push(
                    HeapEntry {
                        score: NotNan::new(sign * root_distances[i]).unwrap(),
                        is_leaf: is_root_leaf as i32,
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

            let distances: Array1<f32> = calculate_distances(
                embeddings_f32,
                &query,
                &self.metric,
            );
            if is_leaf == 1 {
                let children = self.nodes[lvl][node].children().as_ref().unwrap();
                for i in 0..distances.len() {
                    // -1.0 * sign * distance : min sort Vec
                    if !exclude.contains(&children[i]) {
                        items.push((NotNan::new(-1.0 * sign * distances[i]).unwrap(), children[i]));
                    }
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
                if items.len() >= k {
                    items.sort_unstable_by_key(|&(first, _)| first);
                    // println!("Tree_PQ: {:?}, Items: {:?}", tree_pq.len(), items.len());
                    break
                }
                if increments < max_increments || max_increments == -1 {
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
        max_increments: i32,
        exclude: &HashSet<u32>
    ) -> Vec<(NotNan<f32>, u32)> {
        let cnt = self.queries[query_id].items.len().min(k);
        if cnt == 0 && !self.queries[query_id].tree_pq.is_empty() {
            self.incremental_search(query_id, k, search_exp, max_increments, exclude);
        }
        self.queries[query_id].items.drain(0..cnt).collect()
    }
}

#[cfg(test)]
#[path = "utests/ecp_index.rs"]
mod tests;
