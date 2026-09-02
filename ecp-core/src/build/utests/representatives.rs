use super::*;
use std::collections::HashSet;

#[test]
fn offset_strides_by_target_cluster_items() {
    let ids = select_representative_ids(10, 3, RepresentativeStrategy::Offset);
    assert_eq!(ids.to_vec(), vec![0, 3, 6, 9]);
}

#[test]
fn random_returns_distinct_ids_within_range_and_matching_count() {
    let ids = select_representative_ids(100, 10, RepresentativeStrategy::Random);

    assert_eq!(ids.len(), 10, "total_clusters = ceil(100/10)");
    assert!(ids.iter().all(|&id| id < 100));

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
