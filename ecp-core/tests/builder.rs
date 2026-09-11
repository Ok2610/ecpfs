//! Proves `Builder` agrees with search end to end, the same way
//! `tests/build_tree.rs` proves it for the hand-assembled `build_tree`
//! path: build via `Builder`, then load and search with `Index::load`.

use half::f16;
use ndarray::array;
use std::collections::HashSet;
use std::sync::Arc;
use zarrs::array::data_type::{float16, float32};
use zarrs::array::{Array, ArrayBuilder};
use zarrs::filesystem::FilesystemStore;

use ecp_core::build::builder::{Builder, DEFAULT_MAX_CHUNK_BYTES};
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::search::Index;
use ecp_core::utils::{EmbeddingDtype, Metric};

fn write_embeddings(store: &Arc<FilesystemStore>, path: &str, embeddings: &ndarray::Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, float32(), 0.0f32)
        .build(store.clone(), path)
        .expect("failed to build embeddings array");
    array.store_metadata().expect("failed to store embeddings metadata");
    array
        .store_array_subset(&zarrs::array::ArraySubset::new_with_ranges(&[0..embeddings.nrows() as u64, 0..embeddings.ncols() as u64]), embeddings)
        .expect("failed to store embeddings");
}

/// `(node count, total children summed across those nodes)` for one
/// `/lvl_N/` tier.
fn level_node_count_and_children(store: &zarrs::storage::ReadableListableStorage, level: u32) -> (usize, usize) {
    let prefix = zarrs::storage::StorePrefix::new(format!("lvl_{level}/")).unwrap();
    let listing = zarrs::storage::ListableStorageTraits::list_dir(store, &prefix).unwrap();
    let mut total_children = 0;
    for p in listing.prefixes().iter() {
        let emb_path = format!("/{}embeddings", p.as_str());
        let arr = Array::open(store.clone(), &emb_path).expect("node embeddings must exist");
        total_children += arr.shape()[0] as usize;
    }
    (listing.prefixes().len(), total_children)
}

fn write_embeddings_f16(store: &Arc<FilesystemStore>, path: &str, embeddings: &ndarray::Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, float16(), f16::from_f32(0.0))
        .build(store.clone(), path)
        .expect("failed to build embeddings array");
    array.store_metadata().expect("failed to store embeddings metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[0..embeddings.nrows() as u64, 0..embeddings.ncols() as u64]),
            &embeddings.mapv(f16::from_f32),
        )
        .expect("failed to store embeddings");
}

/// Same two-well-separated-clusters-of-4 geometry `tests/build_tree.rs`
/// uses, but selected and built entirely through `Builder`.
#[test]
fn builder_produces_a_structure_that_searches_correctly() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings(
        &store,
        "/dataset",
        &array![
            [0.0f32, 0.0], [0.4, 0.4], [1.0, 1.0], [1.4, 1.4],
            [10.0, 10.0], [10.4, 10.4], [11.0, 11.0], [11.4, 11.4]
        ],
    );
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(&index_path, 2, Metric::L2, false, 1_000_000_000, None, DEFAULT_MAX_CHUNK_BYTES);
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    let root = Array::open(Arc::new(FilesystemStore::new(&index_path).unwrap()), "/index_root/embeddings").unwrap();
    assert_eq!(*root.data_type(), float32(), "f32 source with the default (native) dtype writes f32");

    let mut index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// `ns = node_size`, `total_levels = 3`, chosen so `R = ns^3` exactly (no
/// rounding): 81 items, target_cluster_items=3 -> R=27 representatives,
/// ns=3. Dense ascending 1D values, so every level's leaders partition a
/// contiguous, fully-covered range: no node at any level ends up with zero
/// children. Expected node counts per level: lvl_1 = ns = 3, lvl_2 = ns^2 =
/// 9, lvl_3 (leaf) = ns^3 = R = 27.
#[test]
fn three_level_build_produces_the_right_node_count_per_level_and_searches_to_the_leaf() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    let d = 81;
    let embeddings = ndarray::Array2::from_shape_fn((d, 1), |(i, _)| i as f32);
    write_embeddings(&store, "/dataset", &embeddings);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let ns = 3u32;
    let total_levels = 3u32;
    let target_cluster_items = 3;
    let r = d / target_cluster_items; // 27

    let mut builder = Builder::create(&index_path, total_levels, Metric::L2, false, 1_000_000_000, None, DEFAULT_MAX_CHUNK_BYTES);
    builder.select_representatives(&dataset, target_cluster_items, RepresentativeStrategy::Offset, 1000);
    builder.build(&dataset, 1000);

    let read_store: zarrs::storage::ReadableListableStorage =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to reopen store"));
    for level in 1..=total_levels {
        let (node_count, total_children) = level_node_count_and_children(&read_store, level);
        let expected_nodes = if level == total_levels { r as usize } else { ns.pow(level) as usize };
        let expected_children = if level == total_levels { d } else { ns.pow(level + 1) as usize };
        assert_eq!(node_count, expected_nodes, "lvl_{level} node count");
        assert_eq!(total_children, expected_children, "lvl_{level} total children");
    }

    let mut index = Index::load(index_path, None);
    let query = array![80.0f32];
    let (items, _query_id) = index.new_search(query, 10, 4, -1, &HashSet::new());

    assert!(
        items.iter().any(|(_, id)| *id >= r as u32),
        "search must reach the leaf level and return real dataset item_ids beyond the {r} representatives, got {items:?}"
    );
}

/// A node ending up with fewer children than another (even zero) is
/// expected, not a bug: items with tied nearest-representative scores all
/// route to the same node. Forcing an exact tie (two representatives at
/// the same value) drives this deliberately - lvl_1 ends up with 1 node
/// instead of `ns = 2`, but its total children still add up to `ns^2`.
#[test]
fn a_node_can_end_up_empty_from_tied_scores_without_losing_any_items() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    // Representatives (offset, target_cluster_items=2) are dataset[0,2,4,6]
    // = [0.0, 0.0, 4.0, 6.0]. dataset[2] is forced equal to dataset[0], so
    // root (the first ns=2 representatives) is [0.0, 0.0]: an exact tie.
    let d = 8;
    let embeddings = ndarray::array![[0.0f32], [1.0], [0.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
    write_embeddings(&store, "/dataset", &embeddings);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let ns = 2u32;
    let total_levels = 2u32;
    let target_cluster_items = 2;
    let r = d / target_cluster_items; // 4
    assert_eq!(r, ns.pow(total_levels) as usize, "test setup: r must equal ns^total_levels");

    let mut builder = Builder::create(&index_path, total_levels, Metric::L2, false, 1_000_000_000, None, DEFAULT_MAX_CHUNK_BYTES);
    builder.select_representatives(&dataset, target_cluster_items, RepresentativeStrategy::Offset, 1000);
    builder.build(&dataset, 1000);

    let read_store: zarrs::storage::ReadableListableStorage =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to reopen store"));
    let (lvl_1_nodes, lvl_1_children) = level_node_count_and_children(&read_store, 1);
    assert_eq!(lvl_1_nodes, 1, "the tied root leader gets zero items, so only 1 of ns=2 lvl_1 nodes exists");
    assert_eq!(lvl_1_children, r as usize, "all ns^2 representatives are still accounted for");

    let (lvl_2_nodes, lvl_2_children) = level_node_count_and_children(&read_store, 2);
    assert!(lvl_2_nodes < r as usize, "the tie cascades: lvl_2 also ends up with fewer than ns^2 nodes");
    assert_eq!(lvl_2_children, d, "all D dataset items are still accounted for");
}

/// Regression test: `Index::load` used to build each level's node lookup
/// by listing position rather than by each node's real on-disk id, so as
/// soon as any earlier id in a level was missing (as lvl_1/node_0 is here,
/// same tied-leader setup as the test above), every later id in that level
/// resolved to the wrong slot - out-of-bounds panic or silently the wrong
/// node, depending on how the misalignment landed. This loads the same
/// built index and actually searches it, instead of only inspecting
/// on-disk node/children counts.
#[test]
fn search_still_finds_every_item_when_a_node_is_empty_from_tied_scores() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    let embeddings = ndarray::array![[0.0f32], [1.0], [0.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
    write_embeddings(&store, "/dataset", &embeddings);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(&index_path, 2, Metric::L2, false, 1_000_000_000, None, DEFAULT_MAX_CHUNK_BYTES);
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 1000);
    builder.build(&dataset, 1000);

    let mut index = Index::load(index_path, None);
    let query = array![0.0f32];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let mut ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..8).collect::<Vec<u32>>(), "every item must still be reachable despite the empty lvl_1 node");
}

/// First end-to-end build+search test for `Metric::IP` (previously no
/// coverage anywhere in this suite - see `calculate_distances`, whose IP
/// arm had never been exercised past a single raw-vector unit test).
/// Deliberately uses magnitude-skewed, non-unit vectors: `is_normalized`
/// doesn't affect IP's own assignment or distance math at all (only L2's),
/// so nothing stops a caller from building an IP index on raw, un-normalized
/// embeddings like this. Doing so lets one large-magnitude representative
/// dominate the nearest-representative assignment for nearly every point,
/// which empties out other representatives' nodes - the same failure mode
/// as the tied-score tests above, reached through IP's own math instead of
/// a forced tie.
#[test]
fn ip_metric_builds_and_searches_correctly_even_with_magnitude_skewed_embeddings() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    let embeddings = ndarray::array![
        [1.0f32, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
        [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
    ];
    write_embeddings(&store, "/dataset", &embeddings);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(&index_path, 2, Metric::IP, false, 1_000_000_000, None, DEFAULT_MAX_CHUNK_BYTES);
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    let mut index = Index::load(index_path, None);
    let query = array![1.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let mut ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..8).collect::<Vec<u32>>(), "every item must still be reachable");
    assert_eq!(items[0].1, 3, "item 3 = (4,0) has the highest dot product with the query, so it must rank first");
}

/// Same geometry, but the source is f16-native (as SigLIP-style embeddings
/// often are) and the dtype is left at its default (`None`, native): the
/// index should come out f16 on disk, and still search correctly since
/// `Index::load`/`Node::embeddings` already upcast f16 on read.
#[test]
fn native_dtype_default_writes_f16_when_the_source_is_f16() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings_f16(
        &store,
        "/dataset",
        &array![
            [0.0f32, 0.0], [0.4, 0.4], [1.0, 1.0], [1.4, 1.4],
            [10.0, 10.0], [10.4, 10.4], [11.0, 11.0], [11.4, 11.4]
        ],
    );
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(&index_path, 2, Metric::L2, false, 1_000_000_000, None, DEFAULT_MAX_CHUNK_BYTES);
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    let root = Array::open(Arc::new(FilesystemStore::new(&index_path).unwrap()), "/index_root/embeddings").unwrap();
    assert_eq!(*root.data_type(), float16(), "f16 source with the default (native) dtype writes f16");

    let mut index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// Explicitly requesting `F16` against an f32 source downcasts (with a
/// logged warning); the resulting index is still readable and searchable.
#[test]
fn explicit_f16_downcasts_an_f32_source_and_still_searches() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store = Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings(
        &store,
        "/dataset",
        &array![
            [0.0f32, 0.0], [0.4, 0.4], [1.0, 1.0], [1.4, 1.4],
            [10.0, 10.0], [10.4, 10.4], [11.0, 11.0], [11.4, 11.4]
        ],
    );
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(&index_path, 2, Metric::L2, false, 1_000_000_000, Some(EmbeddingDtype::F16), DEFAULT_MAX_CHUNK_BYTES);
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    let root = Array::open(Arc::new(FilesystemStore::new(&index_path).unwrap()), "/index_root/embeddings").unwrap();
    assert_eq!(*root.data_type(), float16(), "explicit F16 downcasts an f32 source");

    let mut index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}
