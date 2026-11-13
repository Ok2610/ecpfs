use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ordered_float::NotNan;

use super::{Index, persistence};
use crate::error::{EcpError, Result};
use crate::metric::{Metric, calculate_distances};

mod best_first;
mod level_beam;

/// Items a search found, as `(score, item_id)` pairs.
pub type ScoredItems = Vec<(NotNan<f32>, u32)>;

/// The order a search visits the tree's nodes in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Visits the best-scoring queued node next, whatever its level.
    BestFirst,
    /// Visits the `search_exp` best nodes on each level, from the root down. A resumed
    /// search starts at the deepest level holding that many nodes.
    #[default]
    LevelBeam,
}

impl SearchStrategy {
    /// Returns the strategy's name, `"best_first"` or `"level_beam"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchStrategy::BestFirst => "best_first",
            SearchStrategy::LevelBeam => "level_beam",
        }
    }

    /// Returns the strategy named `name`, or `None` for a name [`Self::as_str`] never returns.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "best_first" => Some(SearchStrategy::BestFirst),
            "level_beam" => Some(SearchStrategy::LevelBeam),
            _ => None,
        }
    }
}

/// One open query's search state.
pub(super) struct QueryState {
    pub(super) query: Array1<f32>,
    /// Nodes still to explore.
    pub(super) frontier: Frontier,
    /// Items found but not yet returned, as `(score, item_id)`, best first.
    pub(super) items: ScoredItems,
    /// Item ids left out of every result, the union of each call's `exclude`.
    pub(super) exclude: HashSet<u32>,
}

/// The nodes a query has queued but not yet visited, held the way its search strategy needs.
pub(super) enum Frontier {
    /// One heap across every level, best score first.
    BestFirst(BinaryHeap<HeapEntry>),
    /// One heap per level, indexed by 0-based level, best score first.
    LevelBeam(Vec<BinaryHeap<HeapEntry>>),
}

impl Frontier {
    /// Creates an empty frontier for `strategy` on an index with `levels` levels.
    pub(super) fn new(strategy: SearchStrategy, levels: usize) -> Self {
        match strategy {
            SearchStrategy::BestFirst => Frontier::BestFirst(BinaryHeap::new()),
            SearchStrategy::LevelBeam => Frontier::LevelBeam(vec![BinaryHeap::new(); levels]),
        }
    }

    /// Returns the search strategy this frontier serves.
    pub(super) fn strategy(&self) -> SearchStrategy {
        match self {
            Frontier::BestFirst(_) => SearchStrategy::BestFirst,
            Frontier::LevelBeam(_) => SearchStrategy::LevelBeam,
        }
    }

    /// Number of queued nodes.
    pub(super) fn len(&self) -> usize {
        match self {
            Frontier::BestFirst(heap) => heap.len(),
            Frontier::LevelBeam(heaps) => heaps.iter().map(BinaryHeap::len).sum(),
        }
    }

    /// True when no node is queued.
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over the queued nodes in no particular order.
    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = &HeapEntry> + '_> {
        match self {
            Frontier::BestFirst(heap) => Box::new(heap.iter()),
            Frontier::LevelBeam(heaps) => Box::new(heaps.iter().flat_map(BinaryHeap::iter)),
        }
    }
}

impl QueryState {
    /// Adds `exclude` to the stored set and removes newly excluded ids from `items`.
    fn add_exclude(&mut self, exclude: &HashSet<u32>) {
        let before = self.exclude.len();
        self.exclude.extend(exclude);
        if self.exclude.len() > before {
            let excluded = &self.exclude;
            self.items.retain(|(_, id)| !excluded.contains(id));
        }
    }
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

/// The parts of a query's state that visiting a node reads or fills.
struct Visit<'a> {
    query: &'a Array1<f32>,
    items: &'a mut ScoredItems,
    exclude: &'a HashSet<u32>,
}

/// What [`Index::visit_node`] found at a node.
enum Visited {
    /// The build never wrote the node.
    Unwritten,
    Leaf,
    Internal,
}

const QUERY_NEVER_PERSISTED: &str = "query was never persisted";

/// Rejects a `search_exp` of 0, since a search that visits no node can't find items.
fn check_search_exp(search_exp: u32) -> Result<()> {
    if search_exp == 0 {
        return Err(EcpError::InvalidInput(
            "search_exp must be at least 1".to_string(),
        ));
    }
    Ok(())
}

impl Index {
    /// Gets the state of query `query_id` from the cache, or from disk on a
    /// miss. `Ok(None)` if it is in neither.
    fn get_or_load_query(&self, query_id: usize) -> Result<Option<Arc<Mutex<QueryState>>>> {
        match self.queries.try_get_with(query_id, || {
            // moka can only say "here it is" or "something failed," so a
            // clean "not found" is faked as this one error and unpacked below.
            persistence::load_query(&self.store, query_id, self.levels as usize)?
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

    /// Returns the factor that turns a distance into a heap score, where higher is closer.
    fn score_sign(&self) -> f32 {
        // BinaryHeap pops the largest score first, so negate L2's distance
        // to make the nearest node the largest.
        match self.metric {
            Metric::L2 => -1.0,
            Metric::IP => 1.0,
        }
    }

    /// Scores every root entry against `query` and pushes it onto `heap`.
    fn queue_root(&self, query: &Array1<f32>, heap: &mut BinaryHeap<HeapEntry>) -> Result<()> {
        let sign = self.score_sign();
        let root_distances: Array1<f32> =
            calculate_distances(&self.root, query, &self.metric, self.is_normalized)?;
        // In a 1-level index, root's entries point straight at leaves.
        let is_root_leaf = self.levels == 1;
        for i in 0..root_distances.len() {
            heap.push(HeapEntry {
                score: NotNan::new(sign * root_distances[i]).unwrap(),
                is_leaf: is_root_leaf as i32,
                level: 0,
                node_id: i as u32,
            });
        }
        Ok(())
    }

    /// Scores the node `entry` names against the query. A leaf adds its items to `items`,
    /// except the exclude set; an internal node pushes its children onto `children_into`.
    fn visit_node(
        &self,
        entry: &HeapEntry,
        visit: &mut Visit,
        children_into: &mut BinaryHeap<HeapEntry>,
    ) -> Result<Visited> {
        let &HeapEntry {
            is_leaf,
            level,
            node_id,
            ..
        } = entry;
        let lvl = level as usize;
        let sign = self.score_sign();
        log::trace!("visiting node lvl={lvl} node={node_id} is_leaf={is_leaf}");
        let node = self.node_at(lvl, node_id);
        // Skip an id that was never written to disk
        let embeddings_f32: &Array2<f32> = match node.embeddings()? {
            Some(embs) => embs,
            None => {
                log::debug!("lvl={lvl} node={node_id} has no embeddings on disk, skipping");
                return Ok(Visited::Unwritten);
            }
        };

        let distances: Array1<f32> = calculate_distances(
            embeddings_f32,
            visit.query,
            &self.metric,
            self.is_normalized,
        )?;
        let children = node.children()?.ok_or_else(|| {
            EcpError::Corrupt(format!(
                "{} has embeddings but no children, likely an interrupted write",
                node.group_path
            ))
        })?;
        let visited = if is_leaf == 1 {
            // For a leaf, collect its items. items sorts ascending, so flip
            // the heap's score back to lower-is-better.
            let before = visit.items.len();
            for i in 0..distances.len() {
                if !visit.exclude.contains(&children[i]) {
                    visit
                        .items
                        .push((NotNan::new(-sign * distances[i]).unwrap(), children[i]));
                }
            }
            log::trace!(
                "lvl={lvl} node={node_id}: collected {} of {} items",
                visit.items.len() - before,
                distances.len()
            );
            Visited::Leaf
        } else {
            // For an internal node, queue its children
            let children_are_leaves = level + 1 == self.levels - 1;
            for i in 0..distances.len() {
                children_into.push(HeapEntry {
                    score: NotNan::new(sign * distances[i]).unwrap(),
                    is_leaf: children_are_leaves as i32,
                    level: level + 1,
                    node_id: children[i],
                });
            }
            Visited::Internal
        };

        // Read the node's arrays into cache
        if self.memory_limit_bytes.is_some() {
            self.nodes.insert((lvl, node_id), node.clone());
        }
        Ok(visited)
    }

    /// Searches for `query` and returns the `k` best items as `(score, item_id)` pairs, lowest
    /// score first (L2 distance, or negated inner product for IP), plus the query id. The query
    /// keeps `exclude` and never returns those ids. Other arguments: [`Self::incremental_search`].
    pub fn new_search(
        &self,
        query: Array1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
        strategy: SearchStrategy,
    ) -> Result<(ScoredItems, usize)> {
        check_search_exp(search_exp)?;
        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        if !self.accepting.load(Ordering::SeqCst) {
            return Ok((Vec::new(), query_id));
        }
        log::debug!(
            "new_search: query_id={query_id} k={k} search_exp={search_exp} \
             max_increments={max_increments} exclude={exclude:?} strategy={strategy:?}"
        );
        self.queries.insert(
            query_id,
            Arc::new(Mutex::new(QueryState {
                query,
                frontier: Frontier::new(strategy, self.levels as usize),
                items: Vec::new(),
                exclude: exclude.clone(),
            })),
        );
        let leaf_nodes_scanned =
            self.incremental_search(query_id, k, search_exp, max_increments)?;
        let items = self.drain_items(query_id, k)?;
        log::info!(
            "search: query_id={query_id} k={k} leaf_nodes_scanned={leaf_nodes_scanned} items_returned={}",
            items.len()
        );
        Ok((items, query_id))
    }

    /// Searches further for query `query_id` in its [`SearchStrategy`]'s order, buffering items
    /// except the query's excluded ids. A round visits `search_exp` leaves (BestFirst) or nodes per
    /// level (LevelBeam); while fewer than `k` items are buffered, `search_exp` doubles, at most
    /// `max_increments` times (`-1` for no limit). Returns how many leaves this call visited.
    pub fn incremental_search(
        &self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
    ) -> Result<u32> {
        check_search_exp(search_exp)?;
        // No-op after shutdown or for an unknown query_id
        if !self.accepting.load(Ordering::SeqCst) {
            return Ok(0);
        }
        let Some(state_arc) = self.get_or_load_query(query_id)? else {
            return Ok(0);
        };
        let leaf_cnt = {
            let mut state = state_arc.lock().unwrap();
            let QueryState {
                query,
                frontier,
                items,
                exclude,
            } = &mut *state;
            let mut visit = Visit {
                query,
                items,
                exclude,
            };

            let leaf_cnt = match frontier {
                Frontier::BestFirst(heap) => {
                    self.best_first(query_id, &mut visit, heap, k, search_exp, max_increments)?
                }
                Frontier::LevelBeam(heaps) => {
                    self.level_beam(query_id, &mut visit, heaps, k, search_exp, max_increments)?
                }
            };

            // Sort once, whichever way the search ended
            visit.items.sort_unstable_by_key(|&(first, _)| first);
            leaf_cnt
        };

        // Read the query's updated state into cache
        if self.memory_limit_bytes.is_some() {
            self.queries.insert(query_id, state_arc);
        }
        Ok(leaf_cnt)
    }

    /// Adds `exclude` to query `query_id`'s exclude set, then returns its next `k` items,
    /// searching further first if fewer are buffered; empty for an unknown `query_id` or after
    /// `shutdown`. Other arguments: [`Self::incremental_search`].
    pub fn get_next_k_items(
        &self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> Result<ScoredItems> {
        check_search_exp(search_exp)?;
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
            let mut state = state_arc.lock().unwrap();
            state.add_exclude(exclude);
            log::debug!(
                "get_next_k_items: query_id={query_id} k={k} search_exp={search_exp} \
                 max_increments={max_increments} exclude={exclude:?}"
            );
            state.items.len() < k && !state.frontier.is_empty()
        };
        let leaf_nodes_scanned = if needs_more_search {
            self.incremental_search(query_id, k, search_exp, max_increments)?
        } else {
            0
        };
        let items = self.drain_items(query_id, k)?;
        log::info!(
            "search: query_id={query_id} k={k} leaf_nodes_scanned={leaf_nodes_scanned} items_returned={}",
            items.len()
        );
        Ok(items)
    }

    /// Ends a search so it can't be resumed. Drops the query's state from memory
    /// without saving it and erases any saved copy. A no-op for an unknown `query_id`.
    pub fn end_search(&self, query_id: usize) -> Result<()> {
        self.queries.invalidate(&query_id);
        persistence::erase_query(&self.store, query_id)
    }
}

#[cfg(test)]
#[path = "utests/query.rs"]
mod tests;
