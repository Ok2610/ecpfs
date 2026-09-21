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
use crate::error::{EcpError, Result, ResultExt};
use crate::metric::Metric;
use crate::search::info::{read_info_fields, read_info_u32};
use crate::search::node::Node;

mod persistence;
mod query;

use query::{HeapEntry, QueryState};

/// Share of the memory limit kept for open queries' search state. The rest
/// caches tree nodes read from disk.
const QUERY_CACHE_MEMORY_FRACTION: f64 = 0.05;

type NodeCache = Cache<(usize, u32), Arc<Node>>;
type QueryCache = Cache<usize, Arc<Mutex<QueryState>>>;

/// An eCP index opened from disk. Nodes are read on first visit and cached
/// in memory, along with open queries. Safe to share across threads; a
/// search and an insert only wait for each other on the same leaf.
pub struct Index {
    store: ReadableWritableListableStorage,
    metric: Metric,
    is_normalized: bool,
    /// Node levels below the root; the last one holds the leaves.
    levels: u32,
    /// The root node's embeddings, always in memory.
    root: Array2<f32>,
    /// Nodes read so far, keyed by `(level, node_id)` with `level` 0-based.
    nodes: NodeCache,
    /// Open queries' search state, by query id. A query saved to disk (by
    /// `shutdown` or eviction) is read back on first use.
    queries: QueryCache,
    next_query_id: AtomicUsize,
    memory_limit_bytes: Option<usize>,
    /// False after `shutdown`; searches then return nothing.
    accepting: AtomicBool,
    /// One lock per leaf node id, held only while that leaf is read from disk
    /// or appended to, so a search never reads a leaf in the middle of an
    /// insert's append. zarrs leaves that to its callers.
    leaf_locks: DashMap<u32, Arc<RwLock<()>>>,
    /// The id `insert` gives the next item. Only ever increases; it runs
    /// ahead of `total_items` when a crash stops an insert between taking
    /// ids and writing the items.
    next_item_id: Mutex<u32>,
    /// How many items are stored on disk.
    total_items: Mutex<u32>,
}

impl Index {
    /// Opens the index at `index_path`, reading its `info/*` fields and
    /// root. Other nodes load on first visit. `memory_limit_bytes` caps the
    /// node and query data kept in memory; `None` means no cap.
    pub fn load(index_path: PathBuf, memory_limit_bytes: Option<usize>) -> Result<Self> {
        if !index_path.exists() {
            return Err(EcpError::NotFound(format!(
                "index directory {} does not exist",
                index_path.display()
            )));
        }
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(&index_path).store_err("failed to open store")?);
        Self::load_from_store(store, memory_limit_bytes)
    }

    /// Loads the index from `store` instead of a path. Otherwise the same as `load`.
    /// Note: This function is only split out from `load` for unit tests.
    fn load_from_store(
        store: ReadableWritableListableStorage,
        memory_limit_bytes: Option<usize>,
    ) -> Result<Self> {
        let (levels, metric, is_normalized) = read_info_fields(&store.clone().readable_listable())?;

        let root_array = Array::open(store.clone(), "/index_root/embeddings")
            .store_err("failed to open index_root/embeddings")?;
        let root: Array2<f32> = read_subset_as_f32(
            &root_array,
            &root_array.subset_all(),
            "index_root/embeddings",
        )?;

        let (nodes, queries) = Self::build_caches(memory_limit_bytes, store.clone());

        // Start after the highest saved query id, so no id is reused
        let next_query_id = persistence::query_ids_on_disk(&store)?
            .into_iter()
            .max()
            .map_or(0, |max_id| max_id + 1);

        let readable = store.clone().readable_listable();
        let next_item_id = read_info_u32(&readable, "next_item_id")?;
        let total_items = read_info_u32(&readable, "total_items")?;

        Ok(Index {
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
        })
    }

    /// Builds the node and query caches, uncapped when `memory_limit_bytes`
    /// is `None`. A query evicted from its cache is saved to `store`.
    fn build_caches(
        memory_limit_bytes: Option<usize>,
        store: ReadableWritableListableStorage,
    ) -> (NodeCache, QueryCache) {
        // One node cache for every level, so one capacity caps the whole tree.
        let mut nodes_builder = Cache::builder();
        let mut queries_builder = Cache::builder();

        // With a limit, split it between the caches, sizing each entry in bytes
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

        // moka also notifies when an entry is replaced or removed by hand, so
        // both listeners ask whether the entry really left to free memory.
        let nodes = nodes_builder
            .eviction_listener(|key, node: Arc<Node>, cause| {
                if cause.was_evicted() {
                    log::debug!(
                        "evicted lvl={} node={} cause={cause:?} bytes={}",
                        key.0,
                        key.1,
                        node.resident_bytes()
                    );
                }
            })
            .build();
        // An evicted query goes to disk, so it can still be resumed.
        let queries = queries_builder
            .eviction_listener(move |query_id, state_arc: Arc<Mutex<QueryState>>, cause| {
                if !cause.was_evicted() {
                    return;
                }
                log::debug!("evicting query_id={query_id} cause={cause:?}");
                let state = state_arc.lock().unwrap();
                // No caller is waiting on an eviction, so a failure here can only be logged.
                if let Err(e) = persistence::persist_or_erase(&store, *query_id, &state) {
                    log::error!("failed to persist evicted query_id={query_id}: {e}");
                }
            })
            .build();

        (nodes, queries)
    }

    /// Changes the memory limit without reloading; `None` removes the cap.
    /// It rebuilds both caches, so it suits occasional reconfiguration, not
    /// a call per query.
    pub fn set_memory_limit_bytes(&mut self, memory_limit_bytes: Option<usize>) {
        self.memory_limit_bytes = memory_limit_bytes;
        let (new_nodes, new_queries) = Self::build_caches(memory_limit_bytes, self.store.clone());

        // Move every cached entry over
        for (key, value) in self.nodes.iter() {
            new_nodes.insert(*key, value);
        }
        for (key, value) in self.queries.iter() {
            new_queries.insert(*key, value);
        }

        // Evict now if the new limit is already exceeded
        new_nodes.run_pending_tasks();
        new_queries.run_pending_tasks();
        self.nodes = new_nodes;
        self.queries = new_queries;
        log::debug!(
            "set_memory_limit_bytes: resident now {} bytes, limit={memory_limit_bytes:?}",
            self.resident_bytes()
        );
    }

    /// Gets node `node_id` at 0-based level `lvl`, reading it from disk on a cache miss.
    /// `node_id` may not exist on disk (only nodes that received a child
    /// during build do); `embeddings`/`children` return `Ok(None)` for those.
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
        // Read the node's arrays into cache
        let _ = node.embeddings();
        let _ = node.children();
        // Store before the lock drops, so insert's invalidate() can never
        // land in between and miss this entry.
        self.nodes.insert((lvl, node_id), node.clone());
        node
    }

    /// Returns how many bytes the node cache currently holds.
    fn resident_bytes(&self) -> usize {
        self.nodes.weighted_size() as usize
    }

    /// Stops new searches and saves every open query to disk, so a later
    /// `Index` can resume it; a query with nothing left to return is erased
    /// instead. Safe to call more than once. `Err` if any query fails to save.
    pub fn shutdown(&self) -> Result<()> {
        self.accepting.store(false, Ordering::SeqCst);
        let mut total = 0;
        let mut failed = 0;
        // Count a failed save and go on, so every open query is tried
        for (query_id, state_arc) in self.queries.iter() {
            total += 1;
            let state = state_arc.lock().unwrap();
            if let Err(e) = persistence::persist_or_erase(&self.store, *query_id, &state) {
                log::error!("failed to persist query_id={query_id} on shutdown: {e}");
                failed += 1;
            }
        }
        if failed > 0 {
            return Err(EcpError::Store(format!(
                "failed to persist {failed} of {total} open queries"
            )));
        }
        Ok(())
    }

    /// Erases every query saved to disk before `cutoff_unix_secs` (seconds
    /// since the Unix epoch). Returns how many were erased.
    pub fn cleanup_persisted_queries_older_than(&self, cutoff_unix_secs: u64) -> Result<usize> {
        persistence::cleanup_older_than(&self.store, cutoff_unix_secs)
    }

    /// Adds each row of `embeddings` to its nearest leaf, converted to the
    /// index's dtype, and returns the new ids in row order. Leaves only grow;
    /// the tree is never rebalanced. Safe alongside searches and inserts.
    pub fn insert(&self, embeddings: Array2<f32>) -> Result<std::ops::Range<u32>> {
        if embeddings.nrows() == 0 {
            let next = *self.next_item_id.lock().unwrap();
            return Ok(next..next);
        }

        // Read the dtype and chunk shape from the root; every node shares them
        let root_array = Array::open(self.store.clone(), "/index_root/embeddings")
            .store_err("failed to open index_root/embeddings")?;
        let embedding_dtype = dtype_of_array(&root_array, "index_root/embeddings")?;
        let chunk_shape: Vec<u64> = root_array
            .chunk_shape_usize(&[0, 0])
            .store_err("failed to read index_root/embeddings chunk shape")?
            .into_iter()
            .map(|v| v as u64)
            .collect();

        // A small cache for this call, since it only holds the nodes on the way
        // down to each leaf
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

        // Take the next ids and save the new next_item_id under one lock, so
        // concurrent inserts can't save it out of order. Done before the
        // append, so a crash skips ids instead of reusing them.
        let start = {
            let mut next = self.next_item_id.lock().unwrap();
            let start = *next;
            *next += embeddings.nrows() as u32;
            write_info_u32(&self.store, "next_item_id", *next)?;
            start
        };
        let end = start + embeddings.nrows() as u32;
        let ids = Array1::from_iter(start..end);

        // Append, then drop the stale cached copy of every leaf appended to
        let touched = add_data(&config, &self.root, &embeddings, &ids)?;
        for (level, node_id) in touched {
            // node_at's cache key is 0-based; on-disk level is 1-based.
            let key = (level as usize - 1, node_id);
            log::debug!("dropping the cached copy of lvl={} node={node_id}", key.0);
            self.nodes.invalidate(&key);
        }

        // Counted only once the vectors are on disk
        {
            let mut total = self.total_items.lock().unwrap();
            *total += embeddings.nrows() as u32;
            write_info_u32(&self.store, "total_items", *total)?;
        }

        Ok(start..end)
    }
}

#[cfg(test)]
#[path = "utests/fixtures.rs"]
mod fixtures;

#[cfg(test)]
#[path = "utests/index.rs"]
mod tests;
