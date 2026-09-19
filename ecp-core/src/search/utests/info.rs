use super::*;
use crate::test_fixtures::{
    as_readable_listable, new_memory_store, write_index_info, write_item_counts, write_rep_item_ids,
};
use ndarray::array;

#[test]
fn index_info_reads_all_5_fields_without_loading_the_tree() {
    let store = new_memory_store();
    write_index_info(&store, 2, "IP", true);
    write_item_counts(&store, 1_000);
    write_rep_item_ids(&store, &array![0u32, 5, 10, 15]);

    let info = IndexInfo::load_from_store(as_readable_listable(&store));

    assert_eq!(info.levels, 2);
    assert_eq!(info.metric, Metric::IP);
    assert!(info.is_normalized);
    assert_eq!(info.total_items, 1_000);
    assert_eq!(info.total_representatives, 4);
}
