use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ndarray::Array1;
use ndarray::Array2;

use dashmap::DashMap;
use half::f16;
use moka::sync::Cache;
use zarrs::array::Array;
use zarrs::array::data_type::{float16, float32};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use ordered_float::NotNan;
use std::collections::BinaryHeap;

use crate::build::tree::{BuildConfig, NodeCache as BuildNodeCache, add_data, write_total_items};
use crate::search::node::Node;
use crate::utils::HeapEntry;
use crate::utils::{Metric, calculate_distances};

#[path = "persistence.rs"]
mod persistence;

/// Fraction of `memory_limit_bytes` reserved for in-flight query state,
/// leaving the rest for the node cache. Mirrors `TRACKED_MEMORY_FRACTION`
/// in `build::builder`.
const QUERY_CACHE_MEMORY_FRACTION: f64 = 0.05;

struct QueryState {
    query: Array1<f32>,
    tree_pq: BinaryHeap<HeapEntry>,
    items: Vec<(NotNan<f32>, u32)>,
}

type NodeCache = Cache<(usize, u32), Arc<Node>>;
type QueryCache = Cache<usize, Arc<Mutex<QueryState>>>;

/// A loaded eCP index. Holds the root and each level's nodes lazily, in a
/// concurrent cache capping how many stay resident, and tracks open
/// queries by id for `new_search`/`get_next_k_items`.
///
/// `nodes` is keyed by `(level, node_id)` and populated on first visit, not
/// upfront at load. A node that received zero assigned children during
/// build never gets a directory written, so ids at any given level are
/// generally sparse, not a dense `0..count` range, and a
/// legitimately-referenced id can still turn out to be one of the missing
/// ones (`Node::embeddings`/`children` already handle that by returning
/// `None`).
///
/// `queries` is lazy the same way: `load` only discovers persisted ids,
/// not their content (`get_or_load_query` reads on first use).
///
/// `leaf_locks` and `total_items` back `insert`'s fine-grained concurrency:
/// a leaf's lock is only taken around that leaf's own on-disk write or cold
/// read, so two operations on different leaves never block each other.
///
/// `leaf_locks` is not protecting against concurrent reads/writes of a
/// node's own chunks; zarrs already parallelizes that internally. It
/// guards a different hazard zarrs explicitly leaves to its callers.
/// Two separate calls can still race on the same array, such as a
/// search reading a leaf while an insert appends to it.
pub struct Index {
    store: ReadableWritableListableStorage,
    metric: Metric,
    is_normalized: bool,
    levels: u32,
    root: Array2<f32>,
    nodes: NodeCache,
    queries: QueryCache,
    next_query_id: AtomicUsize,
    memory_limit_bytes: Option<usize>,
    accepting: AtomicBool,
    leaf_locks: DashMap<u32, Arc<RwLock<()>>>,
    total_items: Mutex<u32>,
}

impl Index {
    /// Loads an index from `index_path`, deriving `metric`, `levels`,
    /// `root`, and every level's node paths from the store itself
    /// (`info/levels`, `info/metric`, `index_root/embeddings`, and each
    /// `lvl_N/node_M` group). `memory_limit_bytes` caps how many bytes of
    /// node and query data stay resident.
    pub fn load(index_path: PathBuf, memory_limit_bytes: Option<usize>) -> Self {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(&index_path).expect("Failed to open store"));
        Self::load_from_store(store, memory_limit_bytes)
    }

    fn load_from_store(
        store: ReadableWritableListableStorage,
        memory_limit_bytes: Option<usize>,
    ) -> Self {
        let (levels, metric, is_normalized) = read_info_fields(&store.clone().readable_listable());

        let root_array = Array::open(store.clone(), "/index_root/embeddings")
            .expect("Failed to open index_root/embeddings");
        let root_dtype = root_array.data_type();
        let root: Array2<f32> = if *root_dtype == float32() {
            root_array
                .retrieve_array_subset::<Array2<f32>>(&root_array.subset_all())
                .expect("Failed to retrieve index_root/embeddings")
        } else if *root_dtype == float16() {
            root_array
                .retrieve_array_subset::<Array2<f16>>(&root_array.subset_all())
                .expect("Failed to retrieve index_root/embeddings")
                .mapv(|x| x.to_f32())
        } else {
            panic!(
                "unknown datatype: index_root/embeddings is {root_dtype:?} (use float32 or float16)"
            )
        };

        let (nodes, queries) = Self::build_caches(memory_limit_bytes, store.clone());

        let next_query_id = persistence::query_ids_on_disk(&store)
            .into_iter()
            .max()
            .map_or(0, |max_id| max_id + 1);

        let total_items = read_total_items(&store.clone().readable_listable());

        Index {
            store,
            metric,
            is_normalized,
            levels,
            root,
            nodes,
            queries,
            next_query_id: AtomicUsize::new(next_query_id),
            memory_limit_bytes,
            accepting: AtomicBool::new(true),
            leaf_locks: DashMap::new(),
            total_items: Mutex::new(total_items),
        }
    }

    /// Builds the node and query caches for `memory_limit_bytes`: no limit
    /// when `None`, otherwise split via `QUERY_CACHE_MEMORY_FRACTION`. One
    /// cache spans every level (not one per level) so a single
    /// `max_capacity` caps the whole tree; separate per-level caches would
    /// each get their own capacity instead, multiplying the effective
    /// limit by the level count.
    ///
    /// `queries`'s eviction listener persists (or erases) whatever moka
    /// evicts under memory pressure.
    fn build_caches(
        memory_limit_bytes: Option<usize>,
        store: ReadableWritableListableStorage,
    ) -> (NodeCache, QueryCache) {
        let mut nodes_builder = Cache::builder();
        let mut queries_builder = Cache::builder();

        if let Some(limit) = memory_limit_bytes {
            let query_capacity = (limit as f64 * QUERY_CACHE_MEMORY_FRACTION) as u64;
            let node_capacity = limit as u64 - query_capacity;
            nodes_builder = nodes_builder
                .max_capacity(node_capacity)
                .weigher(|_key: &(usize, u32), node: &Arc<Node>| node.resident_bytes() as u32);
            queries_builder = queries_builder.max_capacity(query_capacity).weigher(
                |_key: &usize, state: &Arc<Mutex<QueryState>>| {
                    let state = state.lock().unwrap();
                    (state.query.len() * size_of::<f32>()
                        + state.tree_pq.len() * size_of::<HeapEntry>()
                        + state.items.len() * size_of::<(NotNan<f32>, u32)>())
                        as u32
                },
            );
        }

        let nodes = nodes_builder
            .eviction_listener(|key, node: Arc<Node>, cause| {
                log::debug!(
                    "evicted lvl={} node={} cause={cause:?} bytes={}",
                    key.0,
                    key.1,
                    node.resident_bytes()
                );
            })
            .build();
        let queries = queries_builder
            .eviction_listener(move |query_id, state_arc: Arc<Mutex<QueryState>>, cause| {
                log::debug!("evicting query_id={query_id} cause={cause:?}");
                let state = state_arc.lock().unwrap();
                persistence::persist_or_erase(&store, *query_id, &state);
            })
            .build();

        (nodes, queries)
    }

    /// Changes the memory limit on an already-loaded index, so a caller can
    /// raise or lower it without reloading. Rebuilds both the node and
    /// query caches from scratch and transfers their existing entries into
    /// the new capacities, which evicts immediately if the new limit is
    /// already exceeded rather than waiting for the next touch.
    ///
    /// This is an administrative/config operation, not a hot-path one: it
    /// rebuilds both caches on every call, so calling it frequently (e.g.
    /// once per query) would defeat the caching it's meant to control.
    pub fn set_memory_limit_bytes(&mut self, memory_limit_bytes: Option<usize>) {
        self.memory_limit_bytes = memory_limit_bytes;
        let (new_nodes, new_queries) = Self::build_caches(memory_limit_bytes, self.store.clone());
        for (key, value) in self.nodes.iter() {
            new_nodes.insert(*key, value);
        }
        for (key, value) in self.queries.iter() {
            new_queries.insert(*key, value);
        }
        new_nodes.run_pending_tasks();
        new_queries.run_pending_tasks();
        self.nodes = new_nodes;
        self.queries = new_queries;
        log::debug!(
            "set_memory_limit_bytes: resident now {} bytes, limit={memory_limit_bytes:?}",
            self.resident_bytes()
        );
    }

    /// The node at `(lvl, node_id)`, reading it from disk on a cache miss.
    /// `node_id` may not exist on disk (only nodes that received a child
    /// during build do); `embeddings`/`children` return `None` for those.
    fn node_at(&self, lvl: usize, node_id: u32) -> Arc<Node> {
        let is_leaf = lvl + 1 == self.levels as usize;
        let child_key = if is_leaf { "item_ids" } else { "node_ids" };

        // Internal Node
        if !is_leaf {
            return self.nodes.get_with((lvl, node_id), || {
                Arc::new(Node::new(
                    self.store.clone().readable_listable(),
                    format!("/lvl_{}/node_{node_id}", lvl + 1),
                    child_key.to_string(),
                ))
            });
        }

        // Leaf node already cached
        if let Some(hit) = self.nodes.get(&(lvl, node_id)) {
            return hit;
        }

        // Acquire lock before reading from disk
        let lock = self
            .leaf_locks
            .entry(node_id)
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone();
        let _guard = lock.read().unwrap();

        // Someone else may have populated it while we waited for the lock.
        if let Some(hit) = self.nodes.get(&(lvl, node_id)) {
            return hit;
        }
        let node = Arc::new(Node::new(
            self.store.clone().readable_listable(),
            format!("/lvl_{}/node_{node_id}", lvl + 1),
            child_key.to_string(),
        ));
        node.embeddings();
        node.children();
        // Store before the lock drops, so insert's invalidate() can never
        // land in between and miss this entry.
        self.nodes.insert((lvl, node_id), node.clone());
        node
    }

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

    /// Bytes currently resident across every cached node, for eviction
    /// accounting.
    fn resident_bytes(&self) -> usize {
        self.nodes.weighted_size() as usize
    }

    /// Stops accepting new work and persists every currently-held query,
    /// erasing one with nothing left worth resuming. Idempotent. `&self`
    /// so it stays callable alongside concurrent readers.
    pub fn shutdown(&self) {
        self.accepting.store(false, Ordering::SeqCst);
        for (query_id, state_arc) in self.queries.iter() {
            let state = state_arc.lock().unwrap();
            persistence::persist_or_erase(&self.store, *query_id, &state);
        }
    }

    /// Erases every persisted query older than `cutoff_unix_secs` (Unix
    /// seconds). Returns how many were erased.
    pub fn cleanup_persisted_queries_older_than(&self, cutoff_unix_secs: u64) -> usize {
        persistence::cleanup_older_than(&self.store, cutoff_unix_secs)
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

                // Re-insert so the weigher re-runs now that `node`'s real
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

    /// Assigns each embedding row the next available id (`total_items..
    /// total_items+embeddings.nrows()`, row order), routes it to its
    /// nearest leaf, and appends it there. No rebalancing. Concurrent
    /// inserts and searches are safe; two operations only serialize when
    /// they land on the same leaf. Returns the assigned id range.
    ///
    /// Not atomic: a panic or crash partway through can leave `total_items`
    /// ahead of what actually landed on disk, permanently skipping the
    /// unwritten ids (never reusing or colliding with one already written).
    pub fn insert(&self, embeddings: Array2<f32>) -> std::ops::Range<u32> {
        if embeddings.nrows() == 0 {
            let total = *self.total_items.lock().unwrap();
            return total..total;
        }

        // Not cached on Index. Read from index_root/embeddings's own
        // on-disk metadata, which every node in the tree shares.
        let root_array = Array::open(self.store.clone(), "/index_root/embeddings")
            .expect("Failed to open index_root/embeddings");
        let root_dtype = root_array.data_type();
        let embedding_dtype = if *root_dtype == float32() {
            crate::utils::EmbeddingDtype::F32
        } else if *root_dtype == float16() {
            crate::utils::EmbeddingDtype::F16
        } else {
            panic!(
                "unknown datatype: index_root/embeddings is {root_dtype:?} (use float32 or float16)"
            )
        };
        let chunk_shape: Vec<u64> = root_array
            .chunk_shape_usize(&[0, 0])
            .expect("Failed to read index_root/embeddings chunk shape")
            .into_iter()
            .map(|v| v as u64)
            .collect();

        // A throwaway cache for this call's own internal-node reads;
        // small since a naive insert only ever touches its own descent
        // path, not the whole tree.
        let cache_capacity_bytes = self.memory_limit_bytes.unwrap_or(usize::MAX) as u64 / 20;
        let node_cache = BuildNodeCache::new(cache_capacity_bytes);
        let config = BuildConfig {
            store: &self.store,
            node_cache: &node_cache,
            target_level: self.levels,
            total_levels: self.levels,
            metric: self.metric,
            is_normalized: self.is_normalized,
            chunk_shape: &chunk_shape,
            embedding_dtype,
            leaf_locks: Some(&self.leaf_locks),
        };

        // Reserving the range and persisting it happen under the same lock
        // so two concurrent inserts can never write total_items out of
        // order (whichever finishes its disk write last would otherwise
        // overwrite the other's larger value).
        let start = {
            let mut total = self.total_items.lock().unwrap();
            let start = *total;
            *total += embeddings.nrows() as u32;
            write_total_items(&self.store, *total);
            start
        };
        let end = start + embeddings.nrows() as u32;
        let ids = Array1::from_iter(start..end);

        let touched = add_data(&config, &self.root, &embeddings, &ids);
        for (level, node_id) in touched {
            // node_at's cache key is 0-based; on-disk level is 1-based.
            self.nodes.invalidate(&(level as usize - 1, node_id));
        }

        start..end
    }
}

/// Reads `info/levels`, `info/metric`, and `info/is_normalized`, the 3
/// fields both `Index::load` and `IndexInfo::load` need.
fn read_info_fields(store: &ReadableListableStorage) -> (u32, Metric, bool) {
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

    let is_normalized_array = Array::open(store.clone(), "/info/is_normalized")
        .expect("Failed to open info/is_normalized");
    let is_normalized: bool = is_normalized_array
        .retrieve_array_subset::<Vec<bool>>(&is_normalized_array.subset_all())
        .expect("Failed to retrieve info/is_normalized")[0];

    (levels, metric, is_normalized)
}

/// Reads `info/total_items`, shared by `IndexInfo::load_from_store` and
/// `Index::load_from_store`.
fn read_total_items(store: &ReadableListableStorage) -> u32 {
    let array =
        Array::open(store.clone(), "/info/total_items").expect("Failed to open info/total_items");
    array
        .retrieve_array_subset::<Vec<u32>>(&array.subset_all())
        .expect("Failed to retrieve info/total_items")[0]
}

/// An index's `info/*` metadata plus its representative count, read without
/// loading the tree. `total_representatives` comes from `/rep_item_ids`'s
/// shape rather than its own field, the same cheap read `Index::load` uses
/// for array shapes elsewhere.
pub struct IndexInfo {
    pub levels: u32,
    pub metric: Metric,
    pub is_normalized: bool,
    pub total_items: u32,
    pub total_representatives: u32,
}

impl IndexInfo {
    /// Loads an index's info fields from `index_path`.
    pub fn load(index_path: PathBuf) -> Self {
        let store: ReadableListableStorage =
            Arc::new(FilesystemStore::new(&index_path).expect("Failed to open store"));
        Self::load_from_store(store)
    }

    fn load_from_store(store: ReadableListableStorage) -> Self {
        let (levels, metric, is_normalized) = read_info_fields(&store);
        let total_items = read_total_items(&store);

        let rep_ids_array =
            Array::open(store.clone(), "/rep_item_ids").expect("Failed to open rep_item_ids");
        let total_representatives = rep_ids_array.shape()[0] as u32;

        IndexInfo {
            levels,
            metric,
            is_normalized,
            total_items,
            total_representatives,
        }
    }
}

#[cfg(test)]
#[path = "utests/index.rs"]
mod tests;
