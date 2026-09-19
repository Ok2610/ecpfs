use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ndarray::Array1;
use ndarray::Array2;

use dashmap::DashMap;
use moka::sync::Cache;
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

use ordered_float::NotNan;

use crate::build::tree::{BuildConfig, NodeCache as BuildNodeCache, add_data};
use crate::build::writer::write_info_u32;
use crate::dtype::{dtype_of_array, read_subset_as_f32};
use crate::metric::Metric;
use crate::search::info::{read_info_fields, read_info_u32};
use crate::search::node::Node;

mod persistence;
mod query;

use query::{HeapEntry, QueryState};

/// Fraction of `memory_limit_bytes` reserved for in-flight query state,
/// leaving the rest for the node cache. Mirrors `TRACKED_MEMORY_FRACTION`
/// in `build::builder`.
const QUERY_CACHE_MEMORY_FRACTION: f64 = 0.05;

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
/// `leaf_locks`, `next_item_id` and `total_items` back `insert`'s
/// fine-grained concurrency: a leaf's lock is only taken around that leaf's
/// own on-disk write or cold read, so two operations on different leaves
/// never block each other.
///
/// `next_item_id` allocates ids and only ever increases; `total_items`
/// counts what is actually stored. They match until a crash mid-insert
/// leaves a reserved range unwritten, which is why they are separate
/// fields rather than one counter serving both roles.
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
    next_item_id: Mutex<u32>,
    total_items: Mutex<u32>,
}

impl Index {
    /// Loads an index from `index_path`, deriving `metric`, `levels` and
    /// `root` from the store itself (`info/levels`, `info/metric`,
    /// `index_root/embeddings`). `memory_limit_bytes` caps how many bytes of
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
        let root: Array2<f32> = read_subset_as_f32(
            &root_array,
            &root_array.subset_all(),
            "index_root/embeddings",
        );

        let (nodes, queries) = Self::build_caches(memory_limit_bytes, store.clone());

        let next_query_id = persistence::query_ids_on_disk(&store)
            .into_iter()
            .max()
            .map_or(0, |max_id| max_id + 1);

        let readable = store.clone().readable_listable();
        let next_item_id = read_info_u32(&readable, "next_item_id");
        let total_items = read_info_u32(&readable, "total_items");

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
            next_item_id: Mutex::new(next_item_id),
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

    /// Assigns each embedding row the next available id (`next_item_id..
    /// next_item_id+embeddings.nrows()`, row order), routes it to its
    /// nearest leaf, and appends it there. No rebalancing. Concurrent
    /// inserts and searches are safe; two operations only serialize when
    /// they land on the same leaf. Returns the assigned id range.
    ///
    /// Not atomic: a panic or crash partway through can leave `next_item_id`
    /// ahead of what actually landed on disk, permanently skipping the
    /// unwritten ids (never reusing or colliding with one already written).
    /// `total_items` is written only once the vectors are stored, so it
    /// keeps counting what exists rather than what was reserved.
    pub fn insert(&self, embeddings: Array2<f32>) -> std::ops::Range<u32> {
        if embeddings.nrows() == 0 {
            let next = *self.next_item_id.lock().unwrap();
            return next..next;
        }

        // Not cached on Index. Read from index_root/embeddings's own
        // on-disk metadata, which every node in the tree shares.
        let root_array = Array::open(self.store.clone(), "/index_root/embeddings")
            .expect("Failed to open index_root/embeddings");
        let embedding_dtype = dtype_of_array(&root_array, "index_root/embeddings");
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
        // so two concurrent inserts can never write next_item_id out of
        // order (whichever finishes its disk write last would otherwise
        // overwrite the other's larger value). Reserved before the write, so
        // a crash skips ids rather than risking a collision.
        let start = {
            let mut next = self.next_item_id.lock().unwrap();
            let start = *next;
            *next += embeddings.nrows() as u32;
            write_info_u32(&self.store, "next_item_id", *next);
            start
        };
        let end = start + embeddings.nrows() as u32;
        let ids = Array1::from_iter(start..end);

        let touched = add_data(&config, &self.root, &embeddings, &ids);
        for (level, node_id) in touched {
            // node_at's cache key is 0-based; on-disk level is 1-based.
            self.nodes.invalidate(&(level as usize - 1, node_id));
        }

        // Counted only now the vectors are on disk. Relative, so two
        // concurrent inserts both land whichever order they finish in.
        {
            let mut total = self.total_items.lock().unwrap();
            *total += embeddings.nrows() as u32;
            write_info_u32(&self.store, "total_items", *total);
        }

        start..end
    }
}

#[cfg(test)]
#[path = "utests/fixtures.rs"]
mod fixtures;

#[cfg(test)]
#[path = "utests/index.rs"]
mod tests;
