use std::collections::BinaryHeap;

use super::{HeapEntry, Visit, Visited};
use crate::error::Result;
use crate::search::index::Index;

impl Index {
    /// Visits nodes from `heap`, best score first, until `search_exp` leaves are scanned. If
    /// fewer than `k` items are in `items` by then, it doubles `search_exp` and goes on, at most
    /// `max_increments` times (`-1` for no limit). Returns how many leaves it visited.
    pub(super) fn best_first(
        &self,
        query_id: usize,
        visit: &mut Visit,
        heap: &mut BinaryHeap<HeapEntry>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
    ) -> Result<u32> {
        let mut search_exp = search_exp;
        let mut increments = 0;
        let mut leaf_cnt = 0;

        // On the first call, queue every root entry
        if heap.is_empty() {
            self.queue_root(visit.query, heap)?;
        }

        while let Some(entry) = heap.pop() {
            match self.visit_node(&entry, visit, heap)? {
                Visited::Unwritten => continue,
                Visited::Leaf => leaf_cnt += 1,
                Visited::Internal => {}
            }

            // After search_exp leaves, stop or double search_exp
            if leaf_cnt == search_exp {
                if visit.items.len() >= k {
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
        }
        Ok(leaf_cnt)
    }
}
