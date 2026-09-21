use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ordered_float::NotNan;

use super::{Index, persistence};
use crate::error::{EcpError, Result};
use crate::metric::{Metric, calculate_distances};

/// Items a search found, as `(score, item_id)` pairs.
pub type ScoredItems = Vec<(NotNan<f32>, u32)>;

/// One open query's search state.
pub(super) struct QueryState {
    pub(super) query: Array1<f32>,
    /// Nodes still to explore, best score first.
    pub(super) tree_pq: BinaryHeap<HeapEntry>,
    /// Items found but not yet returned, as `(score, item_id)`, best first.
    pub(super) items: ScoredItems,
}

/// A node waiting in a search's priority queue, ordered by `score`.
#[derive(Debug, Clone)]
pub(super) struct HeapEntry {
    /// Higher means closer to the query.
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

const QUERY_NEVER_PERSISTED: &str = "query was never persisted";

impl Index {
    /// Gets the state of query `query_id` from the cache, or from disk on a
    /// miss. `Ok(None)` if it is in neither.
    fn get_or_load_query(&self, query_id: usize) -> Result<Option<Arc<Mutex<QueryState>>>> {
        match self.queries.try_get_with(query_id, || {
            // moka can only say "here it is" or "something failed," so a
            // clean "not found" is faked as this one error and unpacked below.
            persistence::load_query(&self.store, query_id)?
                .map(|state| Arc::new(Mutex::new(state)))
                .ok_or_else(|| EcpError::NotFound(QUERY_NEVER_PERSISTED.to_string()))
        }) {
            Ok(state_arc) => Ok(Some(state_arc)),
            Err(e) if matches!(&*e, EcpError::NotFound(msg) if msg == QUERY_NEVER_PERSISTED) => {
                Ok(None)
            }
            Err(e) => Err((*e).clone()),
        }
    }

    /// Removes and returns up to `k` of the best items buffered for query
    /// `query_id`. Empty for an unknown `query_id`.
    fn drain_items(&self, query_id: usize, k: usize) -> Result<ScoredItems> {
        match self.get_or_load_query(query_id)? {
            Some(state_arc) => {
                let mut state = state_arc.lock().unwrap();
                let cnt = state.items.len().min(k);
                Ok(state.items.drain(0..cnt).collect())
            }
            None => Ok(Vec::new()),
        }
    }

    /// Searches for `query` and returns the `k` best items as `(score, item_id)` pairs,
    /// lowest score first (L2 distance, or negated inner product for IP), plus the
    /// query id for `get_next_k_items`. Other arguments: [`Self::incremental_search`].
    pub fn new_search(
        &self,
        query: Array1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> Result<(ScoredItems, usize)> {
        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        if !self.accepting.load(Ordering::SeqCst) {
            return Ok((Vec::new(), query_id));
        }
        log::debug!(
            "new_search: query_id={query_id} k={k} search_exp={search_exp} \
             max_increments={max_increments} exclude={exclude:?}"
        );
        self.queries.insert(
            query_id,
            Arc::new(Mutex::new(QueryState {
                query,
                tree_pq: BinaryHeap::new(),
                items: Vec::new(),
            })),
        );
        let leaves_scanned =
            self.incremental_search(query_id, k, search_exp, max_increments, exclude)?;
        let items = self.drain_items(query_id, k)?;
        log::info!(
            "search: query_id={query_id} k={k} leaves_scanned={leaves_scanned} items_returned={}",
            items.len()
        );
        Ok((items, query_id))
    }

    /// Scores `search_exp` more leaves for query `query_id`, buffering their items
    /// except those in `exclude`. If fewer than `k` items are buffered by then, it doubles
    /// `search_exp` and goes on, at most `max_increments` times (`-1` for no limit).
    /// Returns how many leaves this call visited.
    pub fn incremental_search(
        &self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> Result<u32> {
        // No-op after shutdown or for an unknown query_id
        if !self.accepting.load(Ordering::SeqCst) {
            return Ok(0);
        }
        let Some(state_arc) = self.get_or_load_query(query_id)? else {
            return Ok(0);
        };
        let mut leaf_cnt = 0;
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

            let mut increments = 0;

            // On the first call, queue every root entry
            if tree_pq.is_empty() {
                let root_distances: Array1<f32> =
                    calculate_distances(&self.root, query, &self.metric, self.is_normalized)?;
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
                let embeddings_f32: &Array2<f32> = match node.embeddings()? {
                    Some(embs) => embs,
                    None => {
                        log::warn!("lvl={lvl} node={node_id} has no embeddings on disk, skipping");
                        continue;
                    }
                };

                let distances: Array1<f32> =
                    calculate_distances(embeddings_f32, query, &self.metric, self.is_normalized)?;
                let children = node.children()?.ok_or_else(|| {
                    EcpError::Corrupt(format!(
                        "{} has embeddings but no children, likely an interrupted write",
                        node.group_path
                    ))
                })?;
                if is_leaf == 1 {
                    // For a leaf, collect its items. items sorts ascending, so flip
                    // the heap's score back to lower-is-better.
                    let before = items.len();
                    for i in 0..distances.len() {
                        if !exclude.contains(&children[i]) {
                            items.push((NotNan::new(-sign * distances[i]).unwrap(), children[i]));
                        }
                    }
                    log::trace!(
                        "lvl={lvl} node={node_id}: collected {} of {} items",
                        items.len() - before,
                        distances.len()
                    );
                    leaf_cnt += 1;
                } else {
                    // For an internal node, queue its children
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

                // Read the node's arrays into cache
                if self.memory_limit_bytes.is_some() {
                    self.nodes.insert((lvl, node_id), node.clone());
                }

                // After search_exp leaves, stop or double search_exp
                if leaf_cnt == search_exp {
                    if items.len() >= k {
                        break;
                    }
                    if increments < max_increments || max_increments == -1 {
                        increments += 1;
                        search_exp *= 2;
                        log::debug!(
                            "query_id={query_id}: only {} of {k} items after {leaf_cnt} leaves, \
                             doubling search_exp to {search_exp}",
                            items.len()
                        );
                    } else {
                        break;
                    }
                }
            }

            // Sort once, whichever way the loop ended
            items.sort_unstable_by_key(|&(first, _)| first);
        }

        // Read the query's updated state into cache
        if self.memory_limit_bytes.is_some() {
            self.queries.insert(query_id, state_arc);
        }
        Ok(leaf_cnt)
    }

    /// Returns the next `k` items of query `query_id`, searching further first
    /// if fewer are buffered; empty for an unknown `query_id` or after
    /// `shutdown`. Other arguments: [`Self::incremental_search`].
    pub fn get_next_k_items(
        &self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> Result<ScoredItems> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Ok(Vec::new());
        }
        let Some(state_arc) = self.get_or_load_query(query_id)? else {
            log::warn!(
                "get_next_k_items: query_id={query_id} not found (evicted, invalid, \
                 or never persisted), returning no items"
            );
            return Ok(Vec::new());
        };
        let needs_more_search = {
            let state = state_arc.lock().unwrap();
            log::debug!(
                "get_next_k_items: query_id={query_id} k={k} search_exp={search_exp} \
                 max_increments={max_increments} exclude={exclude:?}"
            );
            state.items.len() < k && !state.tree_pq.is_empty()
        };
        let leaves_scanned = if needs_more_search {
            self.incremental_search(query_id, k, search_exp, max_increments, exclude)?
        } else {
            0
        };
        let items = self.drain_items(query_id, k)?;
        log::info!(
            "search: query_id={query_id} k={k} leaves_scanned={leaves_scanned} items_returned={}",
            items.len()
        );
        Ok(items)
    }
}

#[cfg(test)]
#[path = "utests/query.rs"]
mod tests;
