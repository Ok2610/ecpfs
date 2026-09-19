use super::*;
use crate::dtype::EmbeddingDtype;
use std::collections::HashSet;
use zarrs::array::Array;
use zarrs::storage::store::MemoryStore;

fn is_sorted(ids: &Array1<u32>) -> bool {
    ids.iter().is_sorted()
}

#[test]
fn offset_strides_by_target_cluster_items() {
    let ids = select_representative_ids(10, 3, RepresentativeStrategy::Offset);
    assert_eq!(ids.to_vec(), vec![0, 3, 6, 9]);
}

#[test]
fn random_returns_distinct_sorted_ids_within_range_and_matching_count() {
    let ids = select_representative_ids(100, 10, RepresentativeStrategy::Random);

    assert_eq!(ids.len(), 10, "total_clusters = ceil(100/10)");
    assert!(ids.iter().all(|&id| id < 100));
    assert!(
        is_sorted(&ids),
        "collect_representatives relies on sorted ids for its chunk-skip check"
    );

    let unique: HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "ids must be distinct");
}

/// Uneven division (100/30 doesn't divide evenly) exercises the ceiling in
/// `total_clusters`, matching how many leaders `Offset` produces for the
/// same inputs.
#[test]
fn random_and_offset_agree_on_leader_count_for_uneven_division() {
    let offset_ids = select_representative_ids(100, 30, RepresentativeStrategy::Offset);
    let random_ids = select_representative_ids(100, 30, RepresentativeStrategy::Random);

    assert_eq!(offset_ids.len(), random_ids.len());
}

#[test]
fn fits_in_memory_compares_total_bytes_against_the_limit_inclusively() {
    // 10 vecs * 2 dims * 4 bytes/f32 = 80 bytes.
    assert!(
        fits_in_memory(10, 2, 80),
        "limit equal to the exact byte count must still fit"
    );
    assert!(
        !fits_in_memory(10, 2, 79),
        "one byte under the exact count must not fit"
    );
}

#[test]
fn fits_in_memory_saturates_instead_of_overflow_panicking() {
    assert!(
        !fits_in_memory(usize::MAX, 2, 1_000),
        "saturating_mul must clamp rather than overflow"
    );
}

/// 3 chunks of 2 vecs each (vecs 0-1, 2-3, 4-5); ids 1 and 5 fall in the
/// first and last chunks, leaving the middle chunk (vecs 2-3) unmatched.
fn source_with_skippable_middle_chunk() -> (std::sync::Arc<MemoryStore>, EmbeddingsSource) {
    let store = crate::test_fixtures::new_memory_store();
    let embeddings = ndarray::array![
        [0.0f32, 0.1],
        [10.0, 10.1],
        [20.0, 20.1],
        [30.0, 30.1],
        [40.0, 40.1],
        [50.0, 50.1]
    ];
    let shape = vec![6u64, 2];
    let array = zarrs::array::ArrayBuilder::new(
        shape.clone(),
        vec![2, 2],
        zarrs::array::data_type::float32(),
        0.0f32,
    )
    .build(store.clone(), "/source")
    .expect("failed to build source array");
    array
        .store_metadata()
        .expect("failed to store source metadata");
    array
        .store_array_subset(
            &zarrs::array::ArraySubset::new_with_ranges(&[0..6, 0..2]),
            &embeddings,
        )
        .expect("failed to store source embeddings");

    let source = EmbeddingsSource::Zarr {
        store: crate::test_fixtures::as_readable_listable(&store),
        path: "/source".to_string(),
    };
    (store, source)
}

/// `memory_limit_bytes = 1_000_000` makes `batch_vecs` far exceed the
/// fixture's 6 total items, also proving `end`'s `.min(total_items)` clamp
/// holds; without it, `read_vecs` would panic reading past the end.
#[test]
fn collect_representatives_persists_matched_vecs_and_ids() {
    let (_source_store, source) = source_with_skippable_middle_chunk();
    let dest_store = crate::test_fixtures::new_memory_store();
    let dest = crate::test_fixtures::as_readable_writable_listable(&dest_store);
    let selected_ids = Array1::from_vec(vec![1u32, 5]);

    collect_representatives(
        &dest,
        &source,
        &selected_ids,
        2,
        1_000_000,
        &[100, 2],
        EmbeddingDtype::F32,
    );

    let persisted_ids =
        Array::open(dest.clone(), "/rep_item_ids").expect("rep_item_ids must be persisted");
    assert_eq!(
        persisted_ids
            .retrieve_array_subset::<Array1<u32>>(&persisted_ids.subset_all())
            .unwrap(),
        ndarray::array![1u32, 5]
    );

    let persisted_embeddings =
        Array::open(dest.clone(), "/rep_embeddings").expect("rep_embeddings must be persisted");
    assert_eq!(
        persisted_embeddings
            .retrieve_array_subset::<Array2<f32>>(&persisted_embeddings.subset_all())
            .unwrap(),
        ndarray::array![[10.0f32, 10.1], [50.0, 50.1]]
    );
}

#[test]
/// `memory_limit_bytes = 0` collapses `batch_vecs` to the natural chunk
/// size (2), so this is the test that actually walks all 3 chunks and
/// exercises the middle one's skip branch.
fn collect_representatives_persists_the_same_result_when_batching_skips_a_chunk() {
    let (_source_store, source) = source_with_skippable_middle_chunk();
    let dest_store = crate::test_fixtures::new_memory_store();
    let dest = crate::test_fixtures::as_readable_writable_listable(&dest_store);
    let selected_ids = Array1::from_vec(vec![1u32, 5]);

    collect_representatives(
        &dest,
        &source,
        &selected_ids,
        2,
        0,
        &[100, 2],
        EmbeddingDtype::F32,
    );

    let persisted_embeddings = Array::open(dest.clone(), "/rep_embeddings")
        .expect("rep_embeddings must still be persisted");
    assert_eq!(
        persisted_embeddings
            .retrieve_array_subset::<Array2<f32>>(&persisted_embeddings.subset_all())
            .unwrap(),
        ndarray::array![[10.0f32, 10.1], [50.0, 50.1]]
    );
}
