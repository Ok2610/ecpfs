//! Tests `Builder` end to end. Each test builds an index, then loads it with
//! `Index::load` and searches it.

mod common;

use half::f16;
use ndarray::array;
use std::collections::HashSet;
use std::sync::Arc;
use zarrs::array::data_type::{float16, float32};
use zarrs::array::{Array, ArrayBuilder};
use zarrs::filesystem::FilesystemStore;

use common::{build_index, two_clusters, write_embeddings};
use ecp_core::build::builder::{Builder, DEFAULT_MAX_CHUNK_BYTES};
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::search::Index;
use ecp_core::utils::{EmbeddingDtype, Metric};

/// Counts the nodes on one level (`/lvl_N/`) and their children in total,
/// as `(nodes, children)`.
fn level_node_count_and_children(
    store: &zarrs::storage::ReadableListableStorage,
    level: u32,
) -> (usize, usize) {
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

/// Writes `embeddings` to `path` as a single f16 chunk.
fn write_embeddings_f16(
    store: &Arc<FilesystemStore>,
    path: &str,
    embeddings: &ndarray::Array2<f32>,
) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, float16(), f16::from_f32(0.0))
        .build(store.clone(), path)
        .expect("failed to build embeddings array");
    array
        .store_metadata()
        .expect("failed to store embeddings metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[
                0..embeddings.nrows() as u64,
                0..embeddings.ncols() as u64,
            ]),
            embeddings.mapv(f16::from_f32),
        )
        .expect("failed to store embeddings");
}

#[test]
fn builder_produces_a_structure_that_searches_correctly() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);

    let root = Array::open(
        Arc::new(FilesystemStore::new(&index_path).unwrap()),
        "/index_root/embeddings",
    )
    .unwrap();
    assert_eq!(
        *root.data_type(),
        float32(),
        "f32 source with the default (native) dtype writes f32"
    );

    let index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// A 3-level build sized so `R = ns^3` exactly, with no rounding. 81 items and
/// target_cluster_items=3 give R=27 representatives and ns=3. Dense ascending
/// 1D values, so every level's representatives split a contiguous range and no
/// node at any level ends up with zero children.
///
/// Expected nodes per level: lvl_1 = ns = 3, lvl_2 = ns^2 = 9, lvl_3 (leaf) =
/// ns^3 = R = 27.
#[test]
fn three_level_build_produces_the_right_node_count_per_level_and_searches_to_the_leaf() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    let d = 81;
    let embeddings = ndarray::Array2::from_shape_fn((d, 1), |(i, _)| i as f32);
    write_embeddings(&store, "/dataset", &embeddings);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let ns = 3u32;
    let total_levels = 3u32;
    let target_cluster_items = 3;
    let r = d / target_cluster_items; // 27

    let mut builder = Builder::create(
        &index_path,
        total_levels,
        Metric::L2,
        false,
        1_000_000_000,
        None,
        DEFAULT_MAX_CHUNK_BYTES,
    );
    builder.select_representatives(
        &dataset,
        target_cluster_items,
        RepresentativeStrategy::Offset,
        1000,
    );
    builder.build(&dataset, 1000);

    let read_store: zarrs::storage::ReadableListableStorage =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to reopen store"));
    for level in 1..=total_levels {
        let (node_count, total_children) = level_node_count_and_children(&read_store, level);
        let expected_nodes = if level == total_levels {
            r
        } else {
            ns.pow(level) as usize
        };
        let expected_children = if level == total_levels {
            d
        } else {
            ns.pow(level + 1) as usize
        };
        assert_eq!(node_count, expected_nodes, "lvl_{level} node count");
        assert_eq!(
            total_children, expected_children,
            "lvl_{level} total children"
        );
    }

    let index = Index::load(index_path, None);
    let query = array![80.0f32];
    let (items, _query_id) = index.new_search(query, 10, 4, -1, &HashSet::new());

    assert!(
        items.iter().any(|(_, id)| *id >= r as u32),
        "search must reach the leaf level and return real dataset item_ids beyond the {r} representatives, got {items:?}"
    );
}

/// Tied scores send every tied item to the same node, so a node can end up
/// with no children, which is expected. Two representatives at the same value
/// force a tie. lvl_1 then has 1 node instead of `ns = 2` but still `ns^2`
/// children in total, and search must reach every item without `lvl_1/node_0`.
#[test]
fn a_node_left_empty_by_tied_scores_loses_no_items_on_disk_or_in_search() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    // Representatives (offset, target_cluster_items=2) are dataset[0,2,4,6]
    // = [0.0, 0.0, 4.0, 6.0]. dataset[2] is forced equal to dataset[0], so
    // root (the first ns=2 representatives) is [0.0, 0.0], an exact tie.
    let d = 8;
    let embeddings = ndarray::array![[0.0f32], [1.0], [0.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
    write_embeddings(&store, "/dataset", &embeddings);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let ns = 2u32;
    let total_levels = 2u32;
    let target_cluster_items = 2;
    let r = d / target_cluster_items; // 4
    assert_eq!(
        r,
        ns.pow(total_levels) as usize,
        "test setup: r must equal ns^total_levels"
    );

    let mut builder = Builder::create(
        &index_path,
        total_levels,
        Metric::L2,
        false,
        1_000_000_000,
        None,
        DEFAULT_MAX_CHUNK_BYTES,
    );
    builder.select_representatives(
        &dataset,
        target_cluster_items,
        RepresentativeStrategy::Offset,
        1000,
    );
    builder.build(&dataset, 1000);

    let read_store: zarrs::storage::ReadableListableStorage =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to reopen store"));
    let (lvl_1_nodes, lvl_1_children) = level_node_count_and_children(&read_store, 1);
    assert_eq!(
        lvl_1_nodes, 1,
        "the tied root representative gets zero items, so only 1 of ns=2 lvl_1 nodes exists"
    );
    assert_eq!(
        lvl_1_children, r,
        "all ns^2 representatives are still accounted for"
    );

    let (lvl_2_nodes, lvl_2_children) = level_node_count_and_children(&read_store, 2);
    assert!(
        lvl_2_nodes < r,
        "the tie cascades: lvl_2 also ends up with fewer than ns^2 nodes"
    );
    assert_eq!(
        lvl_2_children, d,
        "all D dataset items are still accounted for"
    );

    let index = Index::load(index_path, None);
    let query = array![0.0f32];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let mut ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..8).collect::<Vec<u32>>(),
        "every item must still be reachable despite the empty lvl_1 node"
    );
}

/// Builds and searches an IP index on vectors of very different magnitudes,
/// which nothing prevents for IP. One large representative then wins nearly
/// every point, leaving other nodes empty, the same situation the tied-score
/// test reaches with a forced tie.
#[test]
fn ip_metric_builds_and_searches_correctly_even_with_magnitude_skewed_embeddings() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    let embeddings = ndarray::array![
        [1.0f32, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [0.0, 4.0]
    ];
    write_embeddings(&store, "/dataset", &embeddings);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(
        &index_path,
        2,
        Metric::IP,
        false,
        1_000_000_000,
        None,
        DEFAULT_MAX_CHUNK_BYTES,
    );
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    let index = Index::load(index_path, None);
    let query = array![1.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let mut ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..8).collect::<Vec<u32>>(),
        "every item must still be reachable"
    );
    assert_eq!(
        items[0].1, 3,
        "item 3 = (4,0) has the highest dot product with the query, so it must rank first"
    );
}

/// An f16 source (as SigLIP embeddings often are) with `embedding_dtype` left
/// as `None` must give an f16 index on disk that still searches correctly.
#[test]
fn native_dtype_default_writes_f16_when_the_source_is_f16() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings_f16(&store, "/dataset", &two_clusters());
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(
        &index_path,
        2,
        Metric::L2,
        false,
        1_000_000_000,
        None,
        DEFAULT_MAX_CHUNK_BYTES,
    );
    builder.select_representatives(&dataset, 2, RepresentativeStrategy::Offset, 100);
    builder.build(&dataset, 100);

    let root = Array::open(
        Arc::new(FilesystemStore::new(&index_path).unwrap()),
        "/index_root/embeddings",
    )
    .unwrap();
    assert_eq!(
        *root.data_type(),
        float16(),
        "f16 source with the default (native) dtype writes f16"
    );

    let index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];
    let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// Integer-valued vectors fit uint8 exactly, so a uint8 build and an f32 build
/// must return identical distances, bit for bit. Widening uint8 to f32 is
/// exact, and the search math is the same from there.
#[test]
fn a_uint8_build_searches_identically_to_the_same_vectors_as_f32() {
    let vectors = array![
        [0.0f32, 0.0],
        [2.0, 2.0],
        [5.0, 5.0],
        [7.0, 7.0],
        [200.0, 200.0],
        [202.0, 202.0],
        [205.0, 205.0],
        [207.0, 207.0]
    ];

    let (_uint8_tmp, uint8_path) = build_index(&vectors, Some(EmbeddingDtype::UInt8));
    let (_f32_tmp, f32_path) = build_index(&vectors, Some(EmbeddingDtype::F32));

    let root = Array::open(
        Arc::new(FilesystemStore::new(&uint8_path).unwrap()),
        "/index_root/embeddings",
    )
    .unwrap();
    assert_eq!(*root.data_type(), zarrs::array::data_type::uint8());

    let query = array![1.0f32, 1.0];
    let (uint8_items, _) =
        Index::load(uint8_path, None).new_search(query.clone(), 8, 4, -1, &HashSet::new());
    let (f32_items, _) = Index::load(f32_path, None).new_search(query, 8, 4, -1, &HashSet::new());

    assert_eq!(
        uint8_items, f32_items,
        "a uint8 index must return the same ids in the same order with the same distances"
    );
    assert_eq!(
        uint8_items.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
}
