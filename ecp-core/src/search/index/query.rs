use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ordered_float::NotNan;

use super::{Index, persistence};
use crate::metric::{Metric, calculate_distances};

pub(super) struct QueryState {
    pub(super) query: Array1<f32>,
    pub(super) tree_pq: BinaryHeap<HeapEntry>,
    pub(super) items: Vec<(NotNan<f32>, u32)>,
}

/// A candidate node in a search's priority queue, ordered by `score`.
#[derive(Debug, Clone)]
pub(super) struct HeapEntry {
    pub(super) score: NotNan<f32>,
    pub(super) is_leaf: i32,
    pub(super) level: u32,
    pub(super) node_id: u32,
}

// We only compare on `score`:
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // forward to `Ord::cmp`
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare only on score:
        self.score.cmp(&other.score)
    }
}

impl Index {
    /// Cache miss falls back to a single-flight disk load; `None` if truly
    /// unknown.
    fn get_or_load_query(&self, query_id: usize) -> Option<Arc<Mutex<QueryState>>> {
        self.queries.optionally_get_with(query_id, || {
            persistence::load_query(&self.store, query_id).map(|state| Arc::new(Mutex::new(state)))
        })
    }

    /// Drains up to `k` ready items from `query_id`'s buffer. A `query_id`
    /// not found anywhere (in memory, on disk, or invalid) yields an empty
    /// result rather than panicking, since that's expected behavior, not a
    /// caller bug.
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

    /// Starts a new query, spending `max_increments` retries on one
    /// `incremental_search` pass, then drains up to `k` items. Returns
    /// `(items, query_id)`. Resume the same query later via
    /// `get_next_k_items` using `query_id`. Each item's score ranks
    /// ascending (lower is better) rather than measuring a literal distance,
    /// since IP's score is a negated similarity where a strong match can be
    /// negative.
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

    /// Descends `tree_pq` (built from `root` on a query's first call),
    /// popping best-scoring entries first: a non-leaf pushes its children,
    /// a leaf accumulates non-excluded candidates into `items`. Once
    /// `search_exp` leaves are explored, stops if `items.len() >= k`, else
    /// doubles `search_exp` and retries (up to `max_increments`, `-1` =
    /// unlimited) before giving up with whatever's found. Mutates the
    /// query's state in place; nothing is returned. A `query_id` not found
    /// is a no-op, same as `drain_items`. Also a no-op after `shutdown`.
    pub fn incremental_search(
        &self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) {
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

            // BinaryHeap only pops the largest score first. IP's similarity is
            // already "higher = better" (sign=1, unchanged); L2's distance is
            // "lower = better", so sign=-1 negates it, making the closest
            // point the largest (least negative) score.
            let sign = match self.metric {
                Metric::L2 => -1.0,
                Metric::IP => 1.0,
            };
            let mut search_exp = search_exp;

            let mut leaf_cnt = 0;
            let mut increments = 0;

            // Add root to tree if empty (new search)
            if tree_pq.is_empty() {
                let root_distances: Array1<f32> =
                    calculate_distances(&self.root, query, &self.metric, self.is_normalized);
                // A 1-level index is IVF-style: node_size == total_clusters, so root
                // already holds every leader and level 0 is the only (leaf) level.
                // Root entries must be marked as leaves from the start in that case,
                // since there is no intermediate level left to descend through.
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
                let embeddings_f32: &Array2<f32> = match node.embeddings() {
                    Some(embs) => embs,
                    None => continue,
                };

                let distances: Array1<f32> =
                    calculate_distances(embeddings_f32, query, &self.metric, self.is_normalized);
                if is_leaf == 1 {
                    let children = node.children().as_ref().unwrap();
                    for i in 0..distances.len() {
                        // items ranks ascending, unlike tree_pq's max-heap, so
                        // the stored score must itself be smaller-is-better;
                        // negating sign again achieves that for both metrics.
                        if !exclude.contains(&children[i]) {
                            items.push((NotNan::new(-sign * distances[i]).unwrap(), children[i]));
                        }
                    }
                    leaf_cnt += 1;
                } else {
                    let children = node.children().as_ref().unwrap();
                    for i in 0..distances.len() {
                        if (level + 1) == (self.levels - 1) {
                            tree_pq.push(HeapEntry {
                                score: NotNan::new(sign * distances[i]).unwrap(),
                                is_leaf: true as i32,
                                level: level + 1,
                                node_id: children[i],
                            });
                        } else {
                            tree_pq.push(HeapEntry {
                                score: NotNan::new(sign * distances[i]).unwrap(),
                                is_leaf: false as i32,
                                level: level + 1,
                                node_id: children[i],
                            });
                        }
                    }
                }

                // Re-insert so the weigher re-runs now that `node`'s
                // size is known (it weighed 0 when `node_at` created it).
                // Re-visits of an already-loaded node don't need this:
                // `get_with` above already counts as an access for moka's
                // recency tracking.
                if self.memory_limit_bytes.is_some() {
                    self.nodes.insert((lvl, node_id), node.clone());
                }

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

            // Every exit above (enough items found, max_increments exhausted,
            // or tree_pq run dry before either) leaves items unsorted; sort
            // once here rather than at each break site.
            items.sort_unstable_by_key(|&(first, _)| first);
        }

        // Same re-weigh reasoning as the node cache: a QueryState grows
        // across its lifetime, so re-insert after mutating it.
        if self.memory_limit_bytes.is_some() {
            self.queries.insert(query_id, state_arc);
        }
    }

    /// Continues `query_id` from where the last call left off: tops up
    /// the buffer with one more `incremental_search` pass if fewer than
    /// `k` items are ready and the tree isn't exhausted, then drains up
    /// to `k`. Same score convention as `new_search`. A `query_id` not
    /// found yields an empty result, same as `drain_items`. Also empty
    /// after `shutdown`.
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
