use super::*;
use crate::search::index::fixtures::{build_test_index, write_ivf_style_fixture};
use crate::test_fixtures::{
    as_readable_listable, as_readable_writable_listable, erase_info_field, new_memory_store,
    write_index_info, write_index_root, write_info_u32, write_item_counts, write_rep_item_ids,
};
use ndarray::array;
use std::collections::HashSet;
use std::sync::Barrier;
use std::thread;

#[test]
fn a_tight_memory_limit_evicts_but_still_searches_correctly() {
    let mut index = build_test_index();
    // Smaller than the ~144 bytes all 6 nodes would take resident at once,
    // but at least as large as the biggest single node (36 bytes), so the
    // node just touched is never itself evicted.
    index.set_memory_limit_bytes(Some(40));
    let query: Array1<f32> = array![0.0, 0.0];

    let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new()).unwrap();

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "eviction must not change search results"
    );
    index.nodes.run_pending_tasks();
    assert!(
        index.resident_bytes() <= 40,
        "resident bytes ({}) exceeded the limit",
        index.resident_bytes()
    );

    let still_present = index.nodes.entry_count();
    assert!(
        still_present < 6,
        "expected eviction to have freed at least one of the 6 touched nodes"
    );
}

#[test]
fn set_memory_limit_bytes_evicts_immediately_if_already_over_the_new_limit() {
    let mut index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];
    index.new_search(query, 4, 4, -1, &HashSet::new()).unwrap();
    index.nodes.run_pending_tasks();
    assert_eq!(
        index.nodes.entry_count(),
        6,
        "sanity check: all 6 nodes loaded with no limit set"
    );

    index.set_memory_limit_bytes(Some(40));

    assert!(
        index.resident_bytes() <= 40,
        "resident bytes ({}) exceeded the limit right after lowering it",
        index.resident_bytes()
    );
    let still_present = index.nodes.entry_count();
    assert!(
        still_present < 6,
        "lowering the limit below current usage must evict immediately, not lazily"
    );
}

/// The fixture's vectors are not unit-length, so an index that fell back to
/// `is_normalized = true` would score every item at distance 1.
#[test]
fn an_index_without_is_normalized_scores_as_not_normalized() {
    let store = write_ivf_style_fixture();
    erase_info_field(&store, "is_normalized");

    let index = Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let (items, _) = index.new_search(query, 2, 4, -1, &HashSet::new()).unwrap();

    let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![0, 1]);
    let expected = (0.4f32 * 0.4 + 0.4 * 0.4).sqrt();
    assert!((items[1].0.into_inner() - expected).abs() < 1e-5);
}

#[test]
fn loading_an_index_without_format_version_adds_it() {
    let store = write_ivf_style_fixture();
    // Every built index has these, and the fixture only writes what a load reads
    write_rep_item_ids(&store, &array![0u32, 2, 4, 6]);
    erase_info_field(&store, "format_version");

    Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();

    let stored = read_info_u32(&as_readable_listable(&store), "format_version").unwrap();
    assert_eq!(stored, FORMAT_VERSION);
}

#[test]
fn loading_an_index_with_an_unknown_format_version_is_refused() {
    let store = write_ivf_style_fixture();
    write_info_u32(&store, "format_version", 2);

    let result = Index::load_from_store(as_readable_writable_listable(&store), None);

    assert!(matches!(result, Err(EcpError::UnsupportedVersion(_))));
}

/// A load that fails must leave the store as it found it, so the missing
/// version is only added once every other read has succeeded.
#[test]
fn a_failed_load_does_not_add_the_missing_format_version() {
    let store = new_memory_store();
    write_index_info(&store, 1, "not a metric", false);
    write_item_counts(&store, 8);
    write_rep_item_ids(&store, &array![0u32, 2]);
    write_index_root(&store, &array![[0.0f32, 0.0]]);
    erase_info_field(&store, "format_version");

    let result = Index::load_from_store(as_readable_writable_listable(&store), None);

    assert!(matches!(result, Err(EcpError::Corrupt(_))));
    assert!(Array::open(store.clone(), "/info/format_version").is_err());
}

#[test]
fn shutdown_then_reload_resumes_a_query_from_a_fresh_index() {
    let store = write_ivf_style_fixture();

    let first_process =
        Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let (first, query_id) = first_process
        .new_search(query, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        first.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    first_process.shutdown().unwrap();
    // A second shutdown must leave the persisted state intact.
    first_process.shutdown().unwrap();

    let second_process =
        Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let second = second_process
        .get_next_k_items(query_id, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        second.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn next_query_id_is_seeded_above_every_persisted_id_after_reload() {
    let store = write_ivf_style_fixture();

    let first_process =
        Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = first_process
        .new_search(query.clone(), 2, 4, -1, &HashSet::new())
        .unwrap();
    first_process.shutdown().unwrap();

    let second_process =
        Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let (_, new_id) = second_process
        .new_search(query, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert!(
        new_id > query_id,
        "new_id ({new_id}) must exceed the reloaded id ({query_id})"
    );
}

#[test]
fn shutdown_blocks_subsequent_calls_on_the_same_instance() {
    let index = build_test_index();
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = index
        .new_search(query.clone(), 2, 4, -1, &HashSet::new())
        .unwrap();
    index.shutdown().unwrap();

    let after_shutdown = index
        .get_next_k_items(query_id, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert!(
        after_shutdown.is_empty(),
        "get_next_k_items must no-op after shutdown"
    );

    let (new_items, new_id) = index.new_search(query, 2, 4, -1, &HashSet::new()).unwrap();
    assert!(new_items.is_empty(), "new_search must no-op after shutdown");
    assert_ne!(new_id, query_id, "a fresh id is still allocated");

    let never_found = index
        .get_next_k_items(new_id, 2, 4, -1, &HashSet::new())
        .unwrap();
    assert!(
        never_found.is_empty(),
        "the post-shutdown id was never actually searched or cached"
    );
}

/// Excluding every item leaves both `tree_pq` and `items` empty once the
/// search ends, so `shutdown` must erase the query.
#[test]
fn exhausted_query_is_erased_not_persisted() {
    let store = write_ivf_style_fixture();

    let index = Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let exclude: HashSet<u32> = (0..8).collect();
    let (items, query_id) = index.new_search(query, 4, 4, -1, &exclude).unwrap();
    assert!(
        items.is_empty(),
        "sanity check: excluding every item must leave nothing found"
    );
    index.shutdown().unwrap();

    assert!(
        persistence::load_query(&index.store, query_id)
            .unwrap()
            .is_none(),
        "an exhausted query must have been erased, not persisted"
    );
}

/// A query stays in memory until it is evicted or `shutdown` saves it, so a
/// search that evicts nothing must leave `/queries/` untouched.
#[test]
fn a_search_alone_does_not_save_its_query_to_disk() {
    let store = write_ivf_style_fixture();

    // Generous enough that nothing is evicted during the search.
    let index =
        Index::load_from_store(as_readable_writable_listable(&store), Some(1 << 20)).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = index.new_search(query, 2, 4, -1, &HashSet::new()).unwrap();
    index.queries.run_pending_tasks();

    assert!(
        persistence::load_query(&index.store, query_id)
            .unwrap()
            .is_none(),
        "a search must not save its query; only eviction or shutdown does"
    );
}

/// The other half of the same rule: a query the cache cannot hold is saved,
/// so it is still resumable.
#[test]
fn a_query_evicted_during_a_search_is_saved_to_disk() {
    let store = write_ivf_style_fixture();

    // The query cache gets 5% of this, well under one QueryState's weight.
    let index = Index::load_from_store(as_readable_writable_listable(&store), Some(400)).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = index.new_search(query, 2, 4, -1, &HashSet::new()).unwrap();
    index.queries.run_pending_tasks();

    assert_eq!(
        index.queries.entry_count(),
        0,
        "sanity check: the query must be evicted"
    );
    assert!(
        persistence::load_query(&index.store, query_id)
            .unwrap()
            .is_some(),
        "an evicted query must be saved so it can be resumed"
    );
}

#[test]
fn evicted_query_resumes_correctly_within_the_same_process() {
    let store = write_ivf_style_fixture();

    let mut index = Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let (first, query_id) = index.new_search(query, 1, 1, -1, &HashSet::new()).unwrap();
    assert_eq!(first.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![0]);

    // Below any QueryState's weight, so the query is evicted right away.
    index.set_memory_limit_bytes(Some(20));
    assert_eq!(
        index.queries.entry_count(),
        0,
        "sanity check: the query must be evicted"
    );

    let second = index
        .get_next_k_items(query_id, 1, 1, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        second.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![1],
        "an evicted query must resume from disk, not come back empty"
    );
}

/// Races `new_search` threads against `shutdown`. A query that starts
/// mid-shutdown may go unpersisted, but nothing may panic, deadlock or
/// persist a torn state.
#[test]
fn shutdown_racing_concurrent_searches_leaves_persisted_state_coherent() {
    let index = Arc::new(build_test_index());
    const THREADS: usize = 8;
    const SEARCHES_PER_THREAD: usize = 20;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let index = Arc::clone(&index);
            std::thread::spawn(move || {
                for _ in 0..SEARCHES_PER_THREAD {
                    let query: Array1<f32> = array![0.0, 0.0];
                    let _ = index.new_search(query, 4, 4, -1, &HashSet::new()).unwrap();
                }
            })
        })
        .collect();

    index.shutdown().unwrap();

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }

    for query_id in 0..(THREADS * SEARCHES_PER_THREAD) {
        if let Some(state) = persistence::load_query(&index.store, query_id).unwrap() {
            let scores: Vec<f32> = state.items.iter().map(|(s, _)| s.into_inner()).collect();
            assert!(
                scores.windows(2).all(|w| w[0] <= w[1]),
                "persisted items for query_id={query_id} are not sorted: {scores:?}"
            );
        }
    }
}

#[test]
fn repersisting_after_more_progress_reflects_the_latest_state() {
    let store = write_ivf_style_fixture();

    let first_process =
        Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let query: Array1<f32> = array![0.0, 0.0];
    let (_, query_id) = first_process
        .new_search(query.clone(), 1, 1, -1, &HashSet::new())
        .unwrap();
    first_process.shutdown().unwrap();

    let second_process =
        Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let drained_more = second_process
        .get_next_k_items(query_id, 2, 1, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        drained_more.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![1, 2]
    );
    second_process.shutdown().unwrap();

    let third_process =
        Index::load_from_store(as_readable_writable_listable(&store), None).unwrap();
    let remaining = third_process
        .get_next_k_items(query_id, 10, 1, -1, &HashSet::new())
        .unwrap();
    assert_eq!(
        remaining.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        vec![3, 4, 5, 6, 7],
        "must continue from the second process's progress, not the first's stale state"
    );
}

#[test]
fn insert_invalidates_exactly_the_touched_cache_entry() {
    let index = build_test_index();
    // search_exp=4 visits all 4 leaves, so every node ends up cached.
    index
        .new_search(array![0.0f32, 0.0], 4, 4, -1, &HashSet::new())
        .unwrap();
    index.nodes.run_pending_tasks();
    let before: Vec<((usize, u32), Arc<Node>)> =
        index.nodes.iter().map(|(key, node)| (*key, node)).collect();
    assert_eq!(before.len(), 6, "sanity check: every node is cached");

    // Close enough to item 0 (in leaf (1, 0)) to route to that exact leaf.
    index.insert(array![[0.01f32, 0.01]]).unwrap();

    assert!(
        index.nodes.get(&(1, 0)).is_none(),
        "the touched leaf's entry must be invalidated"
    );
    for (key, node) in &before {
        if *key == (1, 0) {
            continue;
        }
        let still_there = index
            .nodes
            .get(key)
            .unwrap_or_else(|| panic!("untouched entry {key:?} must not have been evicted"));
        assert!(
            Arc::ptr_eq(node, &still_there),
            "untouched entry {key:?} must keep its original identity"
        );
    }
}

/// Inserts into 4 different leaves from 4 threads released together by a
/// `Barrier`, and checks every insert is found.
#[test]
fn concurrent_inserts_to_different_leaves_all_land() {
    for _ in 0..20 {
        let index = build_test_index();
        let barrier = Barrier::new(4);
        let new_points: [[f32; 2]; 4] =
            [[0.01, 0.01], [1.01, 1.01], [10.01, 10.01], [11.01, 11.01]];

        // Ids depend on which thread reserves first, so record what insert
        // returns.
        let assigned: Vec<(u32, [f32; 2])> = thread::scope(|scope| {
            let handles: Vec<_> = new_points
                .into_iter()
                .map(|point| {
                    let index = &index;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        (index.insert(array![point]).unwrap().start, point)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread panicked"))
                .collect()
        });

        for (id, point) in assigned {
            let (items, _) = index
                .new_search(Array1::from_vec(point.to_vec()), 1, 4, -1, &HashSet::new())
                .unwrap();
            assert_eq!(
                items.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                vec![id],
                "item {id} must be found after concurrent inserts to disjoint leaves"
            );
        }
    }
}

/// One thread searches leaf (1, 0) while another inserts 10 points into it.
/// Every inserted item must be found afterwards, none lost or doubled.
#[test]
fn concurrent_insert_and_search_on_the_same_leaf_do_not_corrupt_data() {
    for _ in 0..20 {
        let index = build_test_index();
        let barrier = Barrier::new(2);

        let assigned_ids: Vec<u32> = thread::scope(|scope| {
            let searcher = &index;
            let searcher_barrier = &barrier;
            scope.spawn(move || {
                searcher_barrier.wait();
                for _ in 0..50 {
                    searcher
                        .new_search(array![0.0f32, 0.0], 10, 4, -1, &HashSet::new())
                        .unwrap();
                }
            });

            let inserter = &index;
            let inserter_barrier = &barrier;
            let handle = scope.spawn(move || {
                inserter_barrier.wait();
                (0..10u32)
                    .map(|i| {
                        inserter
                            .insert(array![[0.01f32 + i as f32 * 0.001, 0.01]])
                            .unwrap()
                            .start
                    })
                    .collect::<Vec<u32>>()
            });
            handle.join().expect("inserter thread panicked")
        });

        let (items, _) = index
            .new_search(array![0.0f32, 0.0], 12, 4, -1, &HashSet::new())
            .unwrap();
        let mut ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        ids.sort_unstable();
        let mut expected: Vec<u32> = assigned_ids;
        expected.extend([0, 1]);
        expected.sort_unstable();
        assert_eq!(
            ids, expected,
            "every concurrently inserted item plus the original leaf contents must be present, with no duplicates or losses"
        );
    }
}
