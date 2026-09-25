use super::*;
use crate::test_fixtures::{
    as_readable_listable, erase_info_field, new_memory_store, write_index_info, write_info_u32,
    write_item_counts, write_rep_item_ids,
};
use ndarray::array;

#[test]
fn index_info_reads_every_field_without_loading_the_tree() {
    let store = new_memory_store();
    write_index_info(&store, 2, "IP", true);
    // Different values, so reading one field from the other's path would show.
    write_info_u32(&store, "total_items", 1_000);
    write_info_u32(&store, "next_item_id", 1_003);
    write_rep_item_ids(&store, &array![0u32, 5, 10, 15]);

    let info = IndexInfo::load_from_store(as_readable_listable(&store)).unwrap();

    assert_eq!(info.levels, 2);
    assert_eq!(info.metric, Metric::IP);
    assert_eq!(info.is_normalized, Some(true));
    assert_eq!(info.total_items, 1_000);
    assert_eq!(info.next_item_id, 1_003);
    assert_eq!(info.total_representatives, 4);
}

/// `is_normalized` is per representation in a future index, so an index that
/// does not record it still loads and reports it as unset.
#[test]
fn index_info_reports_a_missing_is_normalized_as_none() {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", true);
    write_item_counts(&store, 8);
    write_rep_item_ids(&store, &array![0u32, 2]);
    erase_info_field(&store, "is_normalized");

    let info = IndexInfo::load_from_store(as_readable_listable(&store)).unwrap();

    assert_eq!(info.is_normalized, None);
    assert_eq!(info.levels, 2);
}
