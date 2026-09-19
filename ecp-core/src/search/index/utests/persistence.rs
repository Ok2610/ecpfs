use super::*;
use crate::test_fixtures::{as_readable_writable_listable, new_memory_store};
use ndarray::array;

fn heap_entries_sorted(heap: &BinaryHeap<HeapEntry>) -> Vec<(f32, i32, u32, u32)> {
    let mut entries: Vec<_> = heap
        .iter()
        .map(|e| (e.score.into_inner(), e.is_leaf, e.level, e.node_id))
        .collect();
    entries.sort_by(|a, b| a.partial_cmp(b).unwrap());
    entries
}

fn sample_state() -> QueryState {
    QueryState {
        query: array![1.0f32, 2.0, 3.0],
        tree_pq: BinaryHeap::from(vec![
            HeapEntry {
                score: NotNan::new(0.5).unwrap(),
                is_leaf: 0,
                level: 1,
                node_id: 7,
            },
            HeapEntry {
                score: NotNan::new(-1.25).unwrap(),
                is_leaf: 1,
                level: 2,
                node_id: 3,
            },
        ]),
        items: vec![
            (NotNan::new(0.1).unwrap(), 42),
            (NotNan::new(0.9).unwrap(), 99),
        ],
    }
}

#[test]
fn round_trip_preserves_non_empty_state() {
    let store = as_readable_writable_listable(&new_memory_store());
    let original = sample_state();

    persist_query(&store, 5, &original);
    let loaded = load_query(&store, 5).expect("just-persisted query must load");

    assert_eq!(loaded.query, original.query);
    assert_eq!(
        heap_entries_sorted(&loaded.tree_pq),
        heap_entries_sorted(&original.tree_pq)
    );
    assert_eq!(loaded.items, original.items);
}

#[test]
fn round_trip_preserves_fully_empty_state() {
    let store = as_readable_writable_listable(&new_memory_store());
    let original = QueryState {
        query: array![0.0f32, 0.0],
        tree_pq: BinaryHeap::new(),
        items: Vec::new(),
    };

    persist_query(&store, 1, &original);
    let loaded = load_query(&store, 1).expect("just-persisted query must load");

    assert_eq!(loaded.query, original.query);
    assert!(loaded.tree_pq.is_empty());
    assert!(loaded.items.is_empty());
}

#[test]
fn round_trip_preserves_items_only_state() {
    let store = as_readable_writable_listable(&new_memory_store());
    let original = QueryState {
        query: array![0.0f32, 0.0],
        tree_pq: BinaryHeap::new(),
        items: vec![(NotNan::new(0.3).unwrap(), 4)],
    };

    persist_query(&store, 2, &original);
    let loaded = load_query(&store, 2).expect("just-persisted query must load");

    assert!(loaded.tree_pq.is_empty());
    assert_eq!(loaded.items, original.items);
}

#[test]
fn repersisting_with_fewer_entries_drops_the_old_ones() {
    let store = as_readable_writable_listable(&new_memory_store());
    let bigger = sample_state();
    persist_query(&store, 9, &bigger);

    let smaller = QueryState {
        query: array![9.0f32],
        tree_pq: BinaryHeap::new(),
        items: vec![(NotNan::new(1.0).unwrap(), 1)],
    };
    persist_query(&store, 9, &smaller);

    let loaded = load_query(&store, 9).expect("just-persisted query must load");
    assert_eq!(loaded.query, smaller.query);
    assert!(loaded.tree_pq.is_empty());
    assert_eq!(loaded.items, smaller.items);
}

#[test]
fn repersisting_with_more_entries_keeps_all_of_them() {
    let store = as_readable_writable_listable(&new_memory_store());
    let smaller = QueryState {
        query: array![9.0f32],
        tree_pq: BinaryHeap::new(),
        items: vec![(NotNan::new(1.0).unwrap(), 1)],
    };
    persist_query(&store, 9, &smaller);

    let bigger = sample_state();
    persist_query(&store, 9, &bigger);

    let loaded = load_query(&store, 9).expect("just-persisted query must load");
    assert_eq!(loaded.query, bigger.query);
    assert_eq!(
        heap_entries_sorted(&loaded.tree_pq),
        heap_entries_sorted(&bigger.tree_pq)
    );
    assert_eq!(loaded.items, bigger.items);
}

#[test]
fn erasing_a_query_with_no_persisted_group_does_not_panic() {
    let store = as_readable_writable_listable(&new_memory_store());
    erase_query(&store, 123);
}

#[test]
fn loading_a_query_with_no_persisted_group_returns_none() {
    let store = as_readable_writable_listable(&new_memory_store());
    assert!(load_query(&store, 123).is_none());
}

#[test]
fn persist_or_erase_erases_an_empty_state() {
    let store = as_readable_writable_listable(&new_memory_store());
    persist_query(&store, 6, &sample_state());
    assert!(load_query(&store, 6).is_some());

    let empty = QueryState {
        query: array![0.0f32],
        tree_pq: BinaryHeap::new(),
        items: Vec::new(),
    };
    persist_or_erase(&store, 6, &empty);

    assert!(load_query(&store, 6).is_none());
}

#[test]
fn persist_or_erase_persists_a_non_empty_state() {
    let store = as_readable_writable_listable(&new_memory_store());
    let state = sample_state();

    persist_or_erase(&store, 6, &state);

    let loaded = load_query(&store, 6).expect("non-empty state must be persisted, not erased");
    assert_eq!(loaded.items, state.items);
}

#[test]
fn query_ids_on_disk_is_empty_when_the_subtree_does_not_exist() {
    let store = as_readable_writable_listable(&new_memory_store());
    assert!(query_ids_on_disk(&store).is_empty());
}

#[test]
fn query_ids_on_disk_discovers_every_persisted_id() {
    let store = as_readable_writable_listable(&new_memory_store());
    let state = sample_state();
    persist_query(&store, 3, &state);
    persist_query(&store, 7, &state);
    persist_query(&store, 12, &state);

    let mut ids = query_ids_on_disk(&store);
    ids.sort_unstable();
    assert_eq!(ids, vec![3, 7, 12]);
}

#[test]
fn cleanup_older_than_leaves_everything_when_the_cutoff_is_in_the_past() {
    let store = as_readable_writable_listable(&new_memory_store());
    persist_query(&store, 1, &sample_state());
    persist_query(&store, 2, &sample_state());

    let erased = cleanup_older_than(&store, 0);

    assert_eq!(erased, 0);
    assert!(load_query(&store, 1).is_some());
    assert!(load_query(&store, 2).is_some());
}

#[test]
fn cleanup_older_than_erases_only_entries_older_than_the_cutoff() {
    let store = as_readable_writable_listable(&new_memory_store());
    persist_query(&store, 1, &sample_state());
    persist_query(&store, 2, &sample_state());

    // Backdate id=1's persisted_at directly, rather than depending on real
    // elapsed wall-clock time between the two persist_query calls above.
    write_persisted_at(&store, "/queries/1/persisted_at", 1_000);
    let cutoff = 1_600_000_000; // long after id=1's backdated stamp, long before "now"

    let erased = cleanup_older_than(&store, cutoff);

    assert_eq!(erased, 1);
    assert!(
        load_query(&store, 1).is_none(),
        "id=1 was backdated before the cutoff, must be erased"
    );
    assert!(
        load_query(&store, 2).is_some(),
        "id=2 was persisted just now, must survive"
    );
}
