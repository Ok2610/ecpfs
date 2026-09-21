//! Tests `Index::insert` end to end. An inserted point must be found by a
//! later search on the same `Index`, and by a freshly loaded one.

mod common;

use ndarray::array;
use std::collections::HashSet;
use std::sync::Arc;
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;

use common::{build_index, two_clusters, write_embeddings};
use ecp_core::build::builder::{Builder, ChunkSizes};
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::search::{Index, IndexInfo};
use ecp_core::utils::Metric;

#[test]
fn insert_is_found_in_the_same_process_and_after_a_reload() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);
    let index = Index::load(index_path.clone(), None).unwrap();
    let query = array![0.0f32, 0.0];

    // Search once so the leaf covering this region is cached
    let (items, _) = index
        .new_search(query.clone(), 4, 4, -1, &HashSet::new())
        .unwrap();
    assert!(!items.iter().any(|(_, id)| *id == 8));

    // Close enough to item 0 at (0, 0) to land in the same, already cached leaf
    let assigned = index.insert(array![[0.001f32, 0.001]]).unwrap();
    assert_eq!(assigned, 8..9, "the index already has items 0..7");

    let (items, _) = index
        .new_search(query.clone(), 9, 4, -1, &HashSet::new())
        .unwrap();
    assert!(
        items.iter().any(|(_, id)| *id == 8),
        "a stale cached leaf would hide the newly inserted item; got {items:?}"
    );
    drop(index);

    let reloaded = Index::load(index_path, None).unwrap();
    let (items, _) = reloaded
        .new_search(query, 9, 4, -1, &HashSet::new())
        .unwrap();
    assert!(
        items.iter().any(|(_, id)| *id == 8),
        "the newly inserted item must survive a fresh process/Index::load, got {items:?}"
    );
}

/// Overwrites the `/info/{name}` scalar, to set up the counters a crash
/// mid-insert would leave.
fn overwrite_info_u32(index_path: &std::path::Path, name: &str, value: u32) {
    let store: zarrs::storage::ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(index_path).expect("failed to reopen store"));
    let array =
        Array::open(store, &format!("/info/{name}")).expect("failed to open the info field");
    array
        .store_chunk(&[], vec![value])
        .expect("failed to overwrite the info field");
}

/// A crash between taking ids and writing the vectors leaves `next_item_id`
/// ahead of `total_items`. `insert` saves `next_item_id` before writing
/// anything, so a crash in production can leave this state.
#[test]
fn a_reserved_but_unwritten_range_leaves_the_count_honest() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);
    overwrite_info_u32(&index_path, "next_item_id", 20);

    let gapped = IndexInfo::load(index_path.clone()).unwrap();
    assert_eq!(gapped.total_items, 8, "only 8 items were ever written");
    assert_eq!(
        gapped.next_item_id, 20,
        "ids 8..20 were reserved and lost, so they must never be handed out again"
    );

    let index = Index::load(index_path.clone(), None).unwrap();
    let assigned = index.insert(array![[0.05f32, 0.05]]).unwrap();
    assert_eq!(
        assigned,
        20..21,
        "allocation continues from the allocator, never reusing a lost id"
    );
    drop(index);

    let after = IndexInfo::load(index_path).unwrap();
    assert_eq!(
        after.total_items, 9,
        "the count tracks items that exist (8 built + 1 inserted), not the id space"
    );
    assert_eq!(after.next_item_id, 21);
}

#[test]
fn concurrent_inserts_all_land_in_the_count() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);
    let index = Arc::new(Index::load(index_path.clone(), None).unwrap());

    std::thread::scope(|scope| {
        for i in 0..4 {
            let index = Arc::clone(&index);
            scope.spawn(move || {
                let offset = i as f32 * 0.01;
                index
                    .insert(array![[0.05f32 + offset, 0.05], [10.05 + offset, 10.05]])
                    .unwrap();
            });
        }
    });
    drop(index);

    let after = IndexInfo::load(index_path).unwrap();
    assert_eq!(
        after.total_items, 16,
        "8 built plus 4 threads x 2 items; an absolute write would lose whichever finished first"
    );
    assert_eq!(after.next_item_id, 16);
}

#[test]
fn insert_a_batch_routes_each_point_to_its_own_leaf() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);
    let index = Index::load(index_path, None).unwrap();

    let assigned = index
        .insert(array![[0.05f32, 0.05], [10.05, 10.05]])
        .unwrap();
    assert_eq!(assigned, 8..10);

    let (near_origin, _) = index
        .new_search(array![0.0f32, 0.0], 9, 4, -1, &HashSet::new())
        .unwrap();
    assert!(near_origin.iter().any(|(_, id)| *id == 8));

    let (near_far_cluster, _) = index
        .new_search(array![10.0f32, 10.0], 9, 4, -1, &HashSet::new())
        .unwrap();
    assert!(near_far_cluster.iter().any(|(_, id)| *id == 9));
}

/// Places a representative (with `select_representatives_custom`) far from
/// every dataset point, so its leaf is never written during the build.
/// Inserting next to it must create that leaf on disk.
#[test]
fn insert_into_a_previously_empty_leaf_creates_it_on_disk() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = tmp.path().join("index.zarr");
    let store =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to create filesystem store"));

    write_embeddings(&store, "/dataset", &array![[0.0f32], [0.2], [10.0], [10.2]]);
    let dataset = EmbeddingsSource::open(&index_path, "dataset").unwrap();

    let mut builder = Builder::create(
        &index_path,
        1,
        Metric::L2,
        false,
        1_000_000_000,
        None,
        ChunkSizes::default(),
    )
    .unwrap();
    builder
        .select_representatives_custom(array![100u32, 101, 102], array![[0.0f32], [10.0], [1000.0]])
        .unwrap();
    builder.build(&dataset, 100).unwrap();

    let read_store: zarrs::storage::ReadableListableStorage =
        Arc::new(FilesystemStore::new(&index_path).expect("failed to reopen store"));
    assert!(
        Array::open(read_store.clone(), "/lvl_1/node_2/embeddings").is_err(),
        "representative 2 (1000.0) must never have received a dataset point during the initial build"
    );

    let index = Index::load(index_path, None).unwrap();
    let assigned = index.insert(array![[999.0f32]]).unwrap();
    assert_eq!(assigned, 4..5, "the index already has items 0..3");

    let (items, _) = index
        .new_search(array![999.0f32], 1, 4, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        items.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![4],
        "the inserted item must be found via the newly created leaf"
    );
}
