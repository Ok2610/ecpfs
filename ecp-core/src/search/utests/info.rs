use super::*;
use crate::test_fixtures::{
    as_readable_listable, erase_info_field, new_memory_store, write_index_info, write_index_root,
    write_info_u32, write_item_counts, write_rep_item_ids,
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
    assert_eq!(info.format_version, FORMAT_VERSION);
}

/// The version is checked before any other field, so an index from a newer
/// ecpfs reports its version even when that version changed the other fields.
#[test]
fn index_info_refuses_an_unknown_format_version() {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", false);
    write_info_u32(&store, "format_version", 2);

    let Err(EcpError::UnsupportedVersion(message)) =
        IndexInfo::load_from_store(as_readable_listable(&store))
    else {
        panic!("an unknown format version was accepted");
    };

    assert_eq!(
        message,
        format!(
            "index format version 2 is not supported, this build reads version {FORMAT_VERSION}"
        )
    );
}

/// `IndexInfo` opens the store read-only, so it reads the missing field as the
/// current version and leaves the store alone.
#[test]
fn index_info_reads_a_missing_format_version_without_writing_it() {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", false);
    write_item_counts(&store, 8);
    write_rep_item_ids(&store, &array![0u32, 2]);
    write_index_root(&store, &array![[0.0f32, 0.0]]);
    erase_info_field(&store, "format_version");

    let info = IndexInfo::load_from_store(as_readable_listable(&store)).unwrap();

    assert_eq!(info.format_version, FORMAT_VERSION);
    assert!(Array::open(store.clone(), "/info/format_version").is_err());
}

/// The index is repairable only when the version is the one thing missing.
#[test]
fn a_missing_format_version_alongside_other_missing_arrays_names_them_all() {
    let store = new_memory_store();
    write_index_info(&store, 2, "L2", false);
    erase_info_field(&store, "format_version");

    let Err(EcpError::Corrupt(message)) = IndexInfo::load_from_store(as_readable_listable(&store))
    else {
        panic!("an index missing most of its arrays was accepted");
    };

    for missing in [
        "/info/format_version",
        "/info/total_items",
        "/info/next_item_id",
        "/rep_item_ids",
        "/index_root/embeddings",
    ] {
        assert!(
            message.contains(missing),
            "{missing} not named in: {message}"
        );
    }
    assert!(!message.contains("/info/levels"), "{message}");
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
