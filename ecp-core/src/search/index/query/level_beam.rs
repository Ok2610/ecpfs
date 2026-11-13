use std::collections::BinaryHeap;

use super::{HeapEntry, Visit, Visited};
use crate::error::Result;
use crate::search::index::Index;

impl Index {
    /// Descends from the deepest level in `heaps` holding `search_exp` nodes, or from the root,
    /// visiting `search_exp` nodes per level. While fewer than `k` items are in `items`, it doubles
    /// `search_exp` and descends again, at most `max_increments` times (`-1` for no limit).
    /// Returns how many leaves it visited.
    pub(super) fn level_beam(
        &self,
        query_id: usize,
        visit: &mut Visit,
        heaps: &mut [BinaryHeap<HeapEntry>],
        k: usize,
        search_exp: u32,
        max_increments: i32,
    ) -> Result<u32> {
        let mut search_exp = search_exp;
        let mut increments = 0;
        let mut leaf_cnt = 0;

        // On the first call, queue every root entry
        if heaps.iter().all(BinaryHeap::is_empty) {
            self.queue_root(visit.query, &mut heaps[0])?;
        }

        loop {
            // Start at the deepest level holding search_exp nodes, else at the root
            let start = (0..heaps.len())
                .rev()
                .find(|&level| heaps[level].len() >= search_exp as usize)
                .unwrap_or(0);
            leaf_cnt += self.descend(visit, heaps, start, search_exp)?;

            // Stop, or double search_exp and descend again
            if visit.items.len() >= k || heaps.iter().all(BinaryHeap::is_empty) {
                break;
            }
            if increments < max_increments || max_increments == -1 {
                increments += 1;
                search_exp *= 2;
                log::debug!(
                    "query_id={query_id}: only {} of {k} items after {leaf_cnt} leaves, \
                     doubling search_exp to {search_exp}",
                    visit.items.len()
                );
            } else {
                break;
            }
        }
        Ok(leaf_cnt)
    }

    /// Visits up to `search_exp` written nodes from each level's heap, from `start` down to the
    /// leaves. Returns how many leaves it visited.
    fn descend(
        &self,
        visit: &mut Visit,
        heaps: &mut [BinaryHeap<HeapEntry>],
        start: usize,
        search_exp: u32,
    ) -> Result<u32> {
        let mut leaf_cnt = 0;
        for level in start..heaps.len() {
            // Children go to the next level's heap; leaves have none to push
            let (upper, lower) = heaps.split_at_mut(level + 1);
            let heap = &mut upper[level];
            let mut no_children = BinaryHeap::new();
            let children_into = lower.first_mut().unwrap_or(&mut no_children);

            let mut visited = 0;
            while visited < search_exp {
                let Some(entry) = heap.pop() else {
                    break;
                };
                match self.visit_node(&entry, visit, children_into)? {
                    // An unwritten node doesn't use up a slot
                    Visited::Unwritten => continue,
                    Visited::Leaf => leaf_cnt += 1,
                    Visited::Internal => {}
                }
                visited += 1;
            }
        }
        Ok(leaf_cnt)
    }
}
