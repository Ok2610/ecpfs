//! Proves `Index::insert` end to end: a point added after the initial
//! build is durably on disk (found by a freshly reloaded `Index`) and
//! immediately visible within the same process (found by a later search
//! on the same `Index` instance, proving cache invalidation works).

use ndarray::array;
use std::collections::HashSet;
use std::sync::Arc;
use zarrs::array::data_type::float32;
use zarrs::array::{Array, ArrayBuilder};
use zarrs::filesystem::FilesystemStore;

use ecp_core::build::builder::{Builder, DEFAULT_MAX_CHUNK_BYTES};
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::search::{Index, IndexInfo};
use ecp_core::utils::Metric;

fn write_embeddings(store: &Arc<FilesystemStore>, path: &str, embeddings: &ndarray::Array2<f32>) {
    let shape = vec![embeddings.nrows() as u64, embeddings.ncols() as u64];
    let array = ArrayBuilder::new(shape.clone(), shape, float32(), 0.0f32)
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
            embeddings,
        )
        .expect("failed to store embeddings");
}

/// Two well-separated clusters of 4 items each, same geometry
/// `tests/builder.rs`'s `builder_produces_a_structure_that_searches_correctly`
/// uses. Returns the built index's path.
fn build_two_clusters_index() -> std::path::PathBuf {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.keep().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings(
        &store,
        "/dataset",
        &array![
            [0.0f32, 0.0],
            [0.4, 0.4],
            [1.0, 1.0],
            [1.4, 1.4],
            [10.0, 10.0],
            [10.4, 10.4],
            [11.0, 11.0],
            [11.4, 11.4]
        ],
    );
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

    index_path
}

#[test]
fn insert_then_reload_finds_the_new_item_by_search() {
    let index_path = build_two_clusters_index();
    let index = Index::load(index_path.clone(), None);
    let assigned = index.insert(array![[0.05f32, 0.05]]);
    assert_eq!(assigned, 8..9, "the index already has items 0..7");

    let query = array![0.0f32, 0.0];
    let (items, _) = index.new_search(query.clone(), 9, 4, -1, &HashSet::new());
    assert!(
        items.iter().any(|(_, id)| *id == 8),
        "the newly inserted item must be found by the same Index instance, got {items:?}"
    );
    drop(index);

    let reloaded = Index::load(index_path, None);
    let (items, _) = reloaded.new_search(query, 9, 4, -1, &HashSet::new());
    assert!(
        items.iter().any(|(_, id)| *id == 8),
        "the newly inserted item must survive a fresh process/Index::load, got {items:?}"
    );
}

#[test]
fn insert_updates_total_items() {
    let index_path = build_two_clusters_index();
    let before = IndexInfo::load(index_path.clone());
    assert_eq!(before.total_items, 8);

    let index = Index::load(index_path.clone(), None);
    index.insert(array![[0.05f32, 0.05], [10.05, 10.05]]);
    drop(index);

    let after = IndexInfo::load(index_path);
    assert_eq!(after.total_items, 10);
}

#[test]
fn insert_invalidates_the_stale_cached_leaf_within_the_same_process() {
    let index_path = build_two_clusters_index();
    let index = Index::load(index_path, None);
    let query = array![0.0f32, 0.0];

    // Warms whichever leaf serves this region of the tree.
    let (items, _) = index.new_search(query.clone(), 4, 4, -1, &HashSet::new());
    assert!(!items.iter().any(|(_, id)| *id == 8));

    // Close enough to item 0 (at [0,0]) that it must route to the exact
    // same leaf, so this exercises invalidation of an already-cached node.
    index.insert(array![[0.001f32, 0.001]]);

    let (items, _) = index.new_search(query, 9, 4, -1, &HashSet::new());
    assert!(
        items.iter().any(|(_, id)| *id == 8),
        "a stale cached leaf would hide the newly inserted item; got {items:?}"
    );
}

#[test]
fn insert_a_batch_routes_each_point_to_its_own_leaf() {
    let index_path = build_two_clusters_index();
    let index = Index::load(index_path, None);

    let assigned = index.insert(array![[0.05f32, 0.05], [10.05, 10.05]]);
    assert_eq!(assigned, 8..10);

    let (near_origin, _) = index.new_search(array![0.0f32, 0.0], 9, 4, -1, &HashSet::new());
    assert!(near_origin.iter().any(|(_, id)| *id == 8));

    let (near_far_cluster, _) = index.new_search(array![10.0f32, 10.0], 9, 4, -1, &HashSet::new());
    assert!(near_far_cluster.iter().any(|(_, id)| *id == 9));
}

/// Uses `Builder::select_representatives_custom` to place a leader far
/// from every real dataset point, so its leaf is never written during the
/// initial build. Then inserts into exactly that leaf, exercising
/// `zarrs_append`'s "array doesn't exist yet" branch through `insert`.
#[test]
fn insert_into_a_previously_empty_leaf_creates_it_on_disk() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.keep().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings(&store, "/dataset", &array![[0.0f32], [0.2], [10.0], [10.2]]);
    let dataset = EmbeddingsSource::open(&index_path, "dataset");

    let mut builder = Builder::create(
        &index_path,
        1,
        Metric::L2,
        false,
        1_000_000_000,
        None,
        DEFAULT_MAX_CHUNK_BYTES,
    );
    builder.select_representatives_custom(
        array![100u32, 101, 102],
        array![[0.0f32], [10.0], [1000.0]],
    );
    builder.build(&dataset, 100);

    let read_store: zarrs::storage::ReadableListableStorage =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to reopen store"));
    assert!(
        Array::open(read_store.clone(), "/lvl_1/node_2/embeddings").is_err(),
        "leader 2 (1000.0) must never have received a dataset point during the initial build"
    );

    let index = Index::load(index_path, None);
    let assigned = index.insert(array![[999.0f32]]);
    assert_eq!(assigned, 4..5, "the index already has items 0..3");

    let (items, _) = index.new_search(array![999.0f32], 1, 4, -1, &HashSet::new());
    assert_eq!(
        items.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![4],
        "the inserted item must be found via the newly created leaf"
    );
}
