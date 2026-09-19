use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ordered_float::NotNan;

use super::{Index, persistence};
use crate::metric::{Metric, calculate_distances};

/// One open query's search state.
pub(super) struct QueryState {
    pub(super) query: Array1<f32>,
    /// Nodes still to explore, best score first.
    pub(super) tree_pq: BinaryHeap<HeapEntry>,
    /// Items found but not yet returned, sorted best first.
    pub(super) items: Vec<(NotNan<f32>, u32)>,
}

/// A candidate node in a search's priority queue, ordered by `score`.
#[derive(Debug, Clone)]
pub(super) struct HeapEntry {
    pub(super) score: NotNan<f32>,
    /// 1 for a leaf, 0 otherwise.
    pub(super) is_leaf: i32,
    /// 0-based; on disk the node is under `lvl_{level + 1}`.
    pub(super) level: u32,
    pub(super) node_id: u32,
}

// Equality and ordering use `score` only.
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

impl Index {
    /// `query_id`'s state from the cache, or from disk on a miss. `None` if
    /// it is in neither.
    fn get_or_load_query(&self, query_id: usize) -> Option<Arc<Mutex<QueryState>>> {
        self.queries.optionally_get_with(query_id, || {
            persistence::load_query(&self.store, query_id).map(|state| Arc::new(Mutex::new(state)))
        })
    }

    /// Removes and returns up to `k` of `query_id`'s best items. Empty for an
    /// unknown `query_id`.
    fn drain_items(&self, query_id: usize, k: usize) -> Vec<(NotNan<f32>, u32)> {
        match self.get_or_load_query(query_id) {
            Some(state_arc) => {
                let mut state = state_arc.lock().unwrap();
                let cnt = state.items.len().min(k);
                state.items.drain(0..cnt).collect()
            }
            None => Vec::new(),
        }
    }

    /// Starts a query and returns its first `k` items plus its id for
    /// `get_next_k_items`. Lower scores are better (IP negates similarity).
    /// Other arguments work as in [`Self::incremental_search`].
    pub fn new_search(
        &self,
        query: Array1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> (Vec<(NotNan<f32>, u32)>, usize) {
        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        if !self.accepting.load(Ordering::SeqCst) {
            return (Vec::new(), query_id);
        }
        self.queries.insert(
            query_id,
            Arc::new(Mutex::new(QueryState {
                query,
                tree_pq: BinaryHeap::new(),
                items: Vec::new(),
            })),
        );
        self.incremental_search(query_id, k, search_exp, max_increments, exclude);
        (self.drain_items(query_id, k), query_id)
    }

    /// Explores `query_id`'s tree until `search_exp` leaves are scored,
    /// skipping item ids in `exclude`. Below `k` items by then, it doubles
    /// `search_exp` and goes on, at most `max_increments` times (`-1`: no cap).
    pub fn incremental_search(
        &self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) {
        // No-op after shutdown or for an unknown query_id
        if !self.accepting.load(Ordering::SeqCst) {
            return;
        }
        let Some(state_arc) = self.get_or_load_query(query_id) else {
            return;
        };
        {
            let mut state = state_arc.lock().unwrap();
            let QueryState {
                query,
                tree_pq,
                items,
            } = &mut *state;

            // BinaryHeap pops the largest score first, so negate L2's distance
            // to make the nearest node the largest.
            let sign = match self.metric {
                Metric::L2 => -1.0,
                Metric::IP => 1.0,
            };
            let mut search_exp = search_exp;

            let mut leaf_cnt = 0;
            let mut increments = 0;

            // First call: seed tree_pq from root
            if tree_pq.is_empty() {
                let root_distances: Array1<f32> =
                    calculate_distances(&self.root, query, &self.metric, self.is_normalized);
                // In a 1-level index, root's entries point straight at leaves.
                let is_root_leaf = self.levels == 1;
                for i in 0..root_distances.len() {
                    tree_pq.push(HeapEntry {
                        score: NotNan::new(sign * root_distances[i]).unwrap(),
                        is_leaf: is_root_leaf as i32,
                        level: 0,
                        node_id: i as u32,
                    });
                }
            }

            while !tree_pq.is_empty() {
                let HeapEntry {
                    score: _,
                    is_leaf,
                    level,
                    node_id,
                } = tree_pq.pop().unwrap();
                let lvl = level as usize;
                log::trace!("visiting node lvl={lvl} node={node_id} is_leaf={is_leaf}");
                let node = self.node_at(lvl, node_id);
                // Skip an id that was never written to disk
                let embeddings_f32: &Array2<f32> = match node.embeddings() {
                    Some(embs) => embs,
                    None => continue,
                };

                let distances: Array1<f32> =
                    calculate_distances(embeddings_f32, query, &self.metric, self.is_normalized);
                let children = node.children().as_ref().unwrap();
                if is_leaf == 1 {
                    // Leaf: collect its items. items sorts ascending, so flip
                    // the heap's score back to lower-is-better.
                    for i in 0..distances.len() {
                        if !exclude.contains(&children[i]) {
                            items.push((NotNan::new(-sign * distances[i]).unwrap(), children[i]));
                        }
                    }
                    leaf_cnt += 1;
                } else {
                    // Internal: queue its children
                    let children_are_leaves = level + 1 == self.levels - 1;
                    for i in 0..distances.len() {
                        tree_pq.push(HeapEntry {
                            score: NotNan::new(sign * distances[i]).unwrap(),
                            is_leaf: children_are_leaves as i32,
                            level: level + 1,
                            node_id: children[i],
                        });
                    }
                }

                // node_at caches an internal node before its arrays load, at
                // weight 0. Re-insert so moka weighs it again.
                if self.memory_limit_bytes.is_some() {
                    self.nodes.insert((lvl, node_id), node.clone());
                }

                // After search_exp leaves: stop, or double search_exp
                if leaf_cnt == search_exp {
                    if items.len() >= k {
                        break;
                    }
                    if increments < max_increments || max_increments == -1 {
                        increments += 1;
                        search_exp *= 2;
                    } else {
                        break;
                    }
                }
            }

            // Sort once, whichever way the loop ended
            items.sort_unstable_by_key(|&(first, _)| first);
        }

        // Re-insert so moka weighs the state's new size
        if self.memory_limit_bytes.is_some() {
            self.queries.insert(query_id, state_arc);
        }
    }

    /// Returns `query_id`'s next `k` items, searching further first if fewer
    /// are ready; empty for an unknown `query_id` or after `shutdown`. Other
    /// arguments work as in [`Self::incremental_search`].
    pub fn get_next_k_items(
        &self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> Vec<(NotNan<f32>, u32)> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Vec::new();
        }
        let Some(state_arc) = self.get_or_load_query(query_id) else {
            log::debug!(
                "get_next_k_items: query_id={query_id} not found (evicted, invalid, or never persisted), returning no items"
            );
            return Vec::new();
        };
        let needs_more_search = {
            let state = state_arc.lock().unwrap();
            log::debug!(
                "get_next_k_items: query_id={query_id} query={:?} k={k} search_exp={search_exp} max_increments={max_increments} exclude={exclude:?}",
                state.query.to_vec()
            );
            state.items.len() < k && !state.tree_pq.is_empty()
        };
        if needs_more_search {
            self.incremental_search(query_id, k, search_exp, max_increments, exclude);
        }
        self.drain_items(query_id, k)
    }
}

#[cfg(test)]
#[path = "utests/query.rs"]
mod tests;
