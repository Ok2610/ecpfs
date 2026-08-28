use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use ndarray::Array1;

use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorage;

use std::collections::BinaryHeap;
use ordered_float::NotNan;

use crate::ecp_node::Node;
use crate::utils::HeapEntry;
use crate::utils::{calculate_distances, Metric};

struct QueryState {
    query: Array1<f32>,
    tree_pq: BinaryHeap<HeapEntry>,
    items: Vec<(NotNan<f32>, u32)>,
}

pub struct Index {
    metric: Metric,
    levels: u32,
    root: Array2<f32>,
    nodes: Vec<Vec<Node>>,
    queries: Vec<QueryState>,
}

impl Index
{
    /// Creates a new Index instance.
    /// Returns:
    ///     Index<T>: A new instance of Index with the specified metric, levels, root, and nodes.
    pub fn new(
        index_path: PathBuf,
        metric: Metric,
        levels: u32,
        root: Array2<f32>,
        node_paths: Vec<Vec<String>>,
    ) -> Self {
        let mut nodes = Vec::new();
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(&index_path).expect("Failed to open store"));
        for i in 0..levels {
            nodes.push(Vec::new());
            let mut c_key = "node_ids".to_string();
            if i+1 == levels {
                c_key = "item_ids".to_string();
            }
            for node_path in &node_paths[i as usize] {
                let node = Node::new(
                    store.clone(),
                    node_path.clone(),
                    c_key.clone(),
                );
                nodes[i as usize].push(node);
            }
        }
        Index {
            metric: metric.into(),
            levels: levels,
            root: root,
            nodes: nodes,
            queries: Vec::new(),
            // tree_pq: Vec::new(),
            // items: Vec::new(),
        }
    }

    pub fn new_search(
        &mut self,
        query: Array1<f32>,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> (Vec<(NotNan<f32>, u32)>, usize) {
        self.queries.push(QueryState {
            query: query,
            tree_pq: BinaryHeap::new(),
            items: Vec::new()
        });
        // self.tree_pq.push(BinaryHeap::new());
        // self.items.push(Vec::new());
        // self.queries.push(query);
        let query_id = self.queries.len()-1;
        self.incremental_search(query_id, k, search_exp, max_increments, exclude);
        (self.get_next_k_items(query_id, k, search_exp, max_increments, exclude), query_id)
    }

    pub fn incremental_search(
        &mut self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>,
    ) -> () {
        let QueryState{
            query,
            tree_pq,
            items
        }: &mut QueryState = &mut self.queries[query_id];

        // This method will perform an incremental search on the index.
        // It will use the provided query and return the updated priority queues.
        let sign = match self.metric {
            Metric::L2 => -1.0,
            Metric::IP => 1.0,
            Metric::Cos => 1.0,
        };
        let mut search_exp = search_exp;

        let mut leaf_cnt = 0;
        let mut increments = 0;

        // Add root to tree if empty (new search)
        if tree_pq.is_empty() {
            let root_distances: Array1<f32> = calculate_distances(
                &self.root,
                &query,
                &self.metric
            );
            // A 1-level index is IVF-style: node_size == total_clusters, so root
            // already holds every leader and `nodes[0]` is the only (leaf) level.
            // Root entries must be marked as leaves from the start in that case,
            // since there is no intermediate level left to descend through.
            let is_root_leaf = self.levels == 1;
            for i in 0..root_distances.len() {
                tree_pq.push(
                    HeapEntry {
                        score: NotNan::new(sign * root_distances[i]).unwrap(),
                        is_leaf: is_root_leaf as i32,
                        level: 0,
                        node_id: i as u32
                    });
            }
        }

        // Search tree
        while !tree_pq.is_empty() {
            let HeapEntry {
                score: _,
                is_leaf,
                level,
                node_id
            } = tree_pq.pop().unwrap();
            let lvl = level as usize;
            let node = node_id as usize;
            let embeddings_f32: &Array2<f32> = match self.nodes[lvl][node].embeddings() {
                Some(embs) => embs,
                None => continue,
            };

            let distances: Array1<f32> = calculate_distances(
                embeddings_f32,
                &query,
                &self.metric,
            );
            if is_leaf == 1 {
                let children = self.nodes[lvl][node].children().as_ref().unwrap();
                for i in 0..distances.len() {
                    // -1.0 * sign * distance : min sort Vec
                    if !exclude.contains(&children[i]) {
                        items.push((NotNan::new(-1.0 * sign * distances[i]).unwrap(), children[i]));
                    }
                }
                leaf_cnt += 1;
            } else {
                let children = self.nodes[lvl][node].children().as_ref().unwrap();
                for i in 0..distances.len() {
                    // sign * distance : max sort heap queue
                    if (level + 1) == (self.levels - 1) {
                        tree_pq.push(
                            HeapEntry {
                                score: NotNan::new(sign * distances[i]).unwrap(),
                                is_leaf: true as i32,
                                level: level+1,
                                node_id: children[i]
                            });
                    } else {
                        tree_pq.push(
                            HeapEntry {
                                score: NotNan::new(sign * distances[i]).unwrap(),
                                is_leaf: false as i32,
                                level: level + 1,
                                node_id: children[i],
                            });
                    }
                }
            }

            if leaf_cnt == search_exp {
                if items.len() >= k {
                    items.sort_unstable_by_key(|&(first, _)| first);
                    // println!("Tree_PQ: {:?}, Items: {:?}", tree_pq.len(), items.len());
                    break
                }
                if increments < max_increments || max_increments == -1 {
                    increments += 1;
                    search_exp *= 2;
                } else {
                    break
                }
            }
        }
    }

    pub fn get_next_k_items(
        &mut self,
        query_id: usize,
        k: usize,
        search_exp: u32,
        max_increments: i32,
        exclude: &HashSet<u32>
    ) -> Vec<(NotNan<f32>, u32)> {
        let cnt = self.queries[query_id].items.len().min(k);
        if cnt == 0 && !self.queries[query_id].tree_pq.is_empty() {
            self.incremental_search(query_id, k, search_exp, max_increments, exclude);
        }
        self.queries[query_id].items.drain(0..cnt).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{as_readable_listable, new_memory_store, write_node};
    use ndarray::array;

    /// A small 2-level eCP tree, sized and derived the way `ECPBuilder` would for
    /// 8 items with target_cluster_items=2 and levels=2:
    ///   total_clusters = ceil(N / target_cluster_items) = ceil(8 / 2) = 4 leaders
    ///   node_size      = ceil(total_clusters ** (1/levels)) = ceil(sqrt(4)) = 2
    ///
    /// Leaders are picked by striding, exactly like `select_cluster_representatives`'s
    /// default "offset" option (`representative_ids = item_ids[::target_cluster_items]`):
    /// every 2nd item id, i.e. item ids 0, 2, 4, 6. A leader's *global id* (0..3, its
    /// position in that strided list) is what lvl_1's "node_ids" and lvl_2's group
    /// number refer to - it is not the same number as the item id it was struck from.
    ///
    ///   items (id=vector):  0=(0,0)  1=(0.4,0.4)  2=(1,1)  3=(1.4,1.4)
    ///                       4=(10,10) 5=(10.4,10.4) 6=(11,11) 7=(11.4,11.4)
    ///   leaders (global id = item id):  0=item 0   1=item 2   2=item 4   3=item 6
    ///   root:  first node_size (2) leaders -> [leader 0, leader 1]
    ///
    /// lvl_1 buckets every leader under its nearest root leader (striding doesn't
    /// promise a balanced split, so this one naturally comes out 1-vs-3):
    ///   lvl_1/node_0 = {leader 0}            (root leader 0 is its own only member)
    ///   lvl_1/node_1 = {leader 1, 2, 3}
    ///
    /// lvl_2 (leaf) buckets every real item under its nearest leader overall, which
    /// does come out even here (2 items per leader, matching target_cluster_items):
    ///   lvl_2/node_0 = {items 0,1}   lvl_2/node_1 = {items 2,3}
    ///   lvl_2/node_2 = {items 4,5}   lvl_2/node_3 = {items 6,7}
    ///
    /// Nearest-to-farthest from the origin query (0,0) is therefore item id order:
    /// 0, 1, 2, 3, 4, 5, 6, 7.
    fn build_test_index(metric: Metric) -> Index {
        let store = new_memory_store();

        // lvl_1: node_ids are the *global leader ids* assigned to that root leader.
        write_node(
            &store,
            "/lvl_1/node_0",
            &array![[0.0f32, 0.0]], // leader 0 (item 0)
            "node_ids",
            &array![0u32],
        );
        write_node(
            &store,
            "/lvl_1/node_1",
            &array![[1.0f32, 1.0], [10.0, 10.0], [11.0, 11.0]], // leaders 1, 2, 3 (items 2, 4, 6)
            "node_ids",
            &array![1u32, 2, 3],
        );

        // lvl_2: leaf level, so children are "item_ids" (dataset ids) instead.
        // Group numbers 0/1/2/3 line up with leader global ids 0/1/2/3 above.
        write_node(
            &store,
            "/lvl_2/node_0",
            &array![[0.0f32, 0.0], [0.4, 0.4]],
            "item_ids",
            &array![0u32, 1],
        );
        write_node(
            &store,
            "/lvl_2/node_1",
            &array![[1.0f32, 1.0], [1.4, 1.4]],
            "item_ids",
            &array![2u32, 3],
        );
        write_node(
            &store,
            "/lvl_2/node_2",
            &array![[10.0f32, 10.0], [10.4, 10.4]],
            "item_ids",
            &array![4u32, 5],
        );
        write_node(
            &store,
            "/lvl_2/node_3",
            &array![[11.0f32, 11.0], [11.4, 11.4]],
            "item_ids",
            &array![6u32, 7],
        );

        let lvl_1 = vec![
            Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "node_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_1/node_1".to_string(), "node_ids".to_string()),
        ];
        let lvl_2 = vec![
            Node::new(as_readable_listable(&store), "/lvl_2/node_0".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_1".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_2".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_3".to_string(), "item_ids".to_string()),
        ];

        // Constructed as a struct literal (rather than through `Index::new`) so the
        // fixture can use an in-memory store instead of a real `FilesystemStore`.
        Index {
            metric,
            levels: 2,
            root: array![[0.0f32, 0.0], [1.0, 1.0]], // leader 0 (item 0), leader 1 (item 2)
            nodes: vec![lvl_1, lvl_2],
            queries: Vec::new(),
        }
    }

    #[test]
    fn l2_search_returns_nearest_items_in_order() {
        let mut index = build_test_index(Metric::L2);
        let query: Array1<f32> = array![0.0, 0.0];

        // search_exp=4 explores all 4 leaf nodes, so this is an exact top-4.
        let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);

        let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
        assert!(scores.windows(2).all(|w| w[0] <= w[1]), "scores not sorted: {scores:?}");
    }

    #[test]
    fn l2_search_respects_exclude_set() {
        let mut index = build_test_index(Metric::L2);
        let query: Array1<f32> = array![0.0, 0.0];
        let exclude: HashSet<u32> = [0].into_iter().collect();

        let (items, _query_id) = index.new_search(query, 4, 4, -1, &exclude);

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4], "excluded item 0 must not appear");
    }

    #[test]
    fn incremental_search_resumes_and_drains_remaining_items() {
        let mut index = build_test_index(Metric::L2);
        let query: Array1<f32> = array![0.0, 0.0];

        // First page: nearest 2 items.
        let (first, query_id) = index.new_search(query, 2, 4, -1, &HashSet::new());
        assert_eq!(first.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![0, 1]);

        // Second page: continues from where the first left off, same query_id.
        let second = index.get_next_k_items(query_id, 2, 4, -1, &HashSet::new());
        assert_eq!(second.iter().map(|(_, id)| *id).collect::<Vec<_>>(), vec![2, 3]);
    }

    /// With search_exp=1, the first pass only explores 1 leaf cluster (2 items:
    /// ids 0,1) - not enough for k=4. With max_increments=-1 (unlimited), the
    /// retry path at ecp_index.rs's `leaf_cnt == search_exp` check must double
    /// search_exp (1 -> 2) and keep going, exploring a 2nd cluster (ids 2,3) to
    /// reach k. Never exercised before: every existing fixture used a search_exp
    /// large enough to satisfy k on the first pass.
    #[test]
    fn search_exp_doubles_until_k_items_found_with_unlimited_retries() {
        let mut index = build_test_index(Metric::L2);
        let query: Array1<f32> = array![0.0, 0.0];

        let (items, _query_id) = index.new_search(query, 4, 1, -1, &HashSet::new());

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3], "doubling search_exp should eventually surface all 4 nearest items");
    }

    /// Same setup as the unlimited-retry test above, but with a *finite*
    /// max_increments=1 - i.e. "at most 1 retry" - which should be just enough
    /// to go from search_exp=1 to 2 and reach k=4, identically to the unlimited
    /// case. This isolates the finite-counter comparison (`increments >
    /// max_increments`) from the `max_increments == -1` special case, which is
    /// the only branch the test above exercises.
    #[test]
    fn finite_max_increments_still_allows_configured_number_of_retries() {
        let mut index = build_test_index(Metric::L2);
        let query: Array1<f32> = array![0.0, 0.0];

        let (items, _query_id) = index.new_search(query, 4, 1, 1, &HashSet::new());

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2, 3],
            "max_increments=1 should permit the single retry needed to reach k=4"
        );
    }

    /// Same fixture, but k=8 (all items) needs 2 doublings (search_exp 1 -> 2 -> 4)
    /// to satisfy, while max_increments=1 permits only 1. The search must stop
    /// once its retry budget is exhausted rather than looping until k is met -
    /// returning fewer than k items (the 2 clusters/4 items reachable after the
    /// single permitted retry), not all 8.
    #[test]
    fn search_stops_once_max_increments_is_exhausted() {
        let mut index = build_test_index(Metric::L2);
        let query: Array1<f32> = array![0.0, 0.0];

        let (items, _query_id) = index.new_search(query, 8, 1, 1, &HashSet::new());

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2, 3],
            "should stop after its 1 permitted retry (2 clusters explored), not find all 8 items"
        );
    }

    /// A levels=1 index is IVF-style: since node_size = ceil(total_clusters**1) =
    /// total_clusters, the root holds *every* leader directly, and the single
    /// level (nodes[0]) holds the leaf clusters - there is no intermediate level
    /// to descend through. Same 8 items/4 leaders as `build_test_index`, just
    /// flattened: root = all 4 leaders, and each leader's cluster is looked up
    /// directly by its global id.
    fn build_ivf_style_index(metric: Metric) -> Index {
        let store = new_memory_store();

        write_node(&store, "/lvl_1/node_0", &array![[0.0f32, 0.0], [0.4, 0.4]], "item_ids", &array![0u32, 1]);
        write_node(&store, "/lvl_1/node_1", &array![[1.0f32, 1.0], [1.4, 1.4]], "item_ids", &array![2u32, 3]);
        write_node(&store, "/lvl_1/node_2", &array![[10.0f32, 10.0], [10.4, 10.4]], "item_ids", &array![4u32, 5]);
        write_node(&store, "/lvl_1/node_3", &array![[11.0f32, 11.0], [11.4, 11.4]], "item_ids", &array![6u32, 7]);

        let leaf_clusters = vec![
            Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_1/node_1".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_1/node_2".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_1/node_3".to_string(), "item_ids".to_string()),
        ];

        Index {
            metric,
            levels: 1,
            root: array![[0.0f32, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]], // all 4 leaders
            nodes: vec![leaf_clusters],
            queries: Vec::new(),
        }
    }

    /// A 2-level tree like `build_test_index`, but sized for `Metric::Cos` instead
    /// of `Metric::L2`: cosine similarity is undefined for a zero vector (division
    /// by zero) and degenerate for collinear vectors (identical similarity), both
    /// of which `build_test_index`'s geometry hits (item 0 is the origin; items
    /// 1-7 all lie on the same ray). So this fixture uses 8 unit-ish vectors at
    /// distinct angles from the x-axis instead, giving each item a distinct,
    /// hand-computable cosine similarity against the query (1, 0):
    ///
    ///   item id : angle : cos(angle) (= expected similarity to query (1,0))
    ///       0    :   0°  :  1.0
    ///       1    :  20°  :  0.9396926
    ///       2    :  45°  :  0.7071068
    ///       3    :  60°  :  0.5
    ///       4    :  90°  :  0.0
    ///       5    : 120°  : -0.5
    ///       6    : 150°  : -0.8660254
    ///       7    : 180°  : -1.0
    ///
    /// Tree topology mirrors `build_test_index` (root -> lvl_1 -> lvl_2 leaves,
    /// items 0,1 / 2,3 / 4,5 / 6,7 grouped one pair per leaf), with each leader's
    /// embedding reused from the first item in its cluster (so nothing is zero).
    fn build_cos_test_index() -> Index {
        let store = new_memory_store();

        write_node(&store, "/lvl_1/node_0", &array![[1.0f32, 0.0]], "node_ids", &array![0u32]);
        write_node(
            &store,
            "/lvl_1/node_1",
            &array![[1.0f32, 1.0], [0.0, 1.0], [-0.8660254, 0.5]],
            "node_ids",
            &array![1u32, 2, 3],
        );

        write_node(&store, "/lvl_2/node_0", &array![[1.0f32, 0.0], [0.9396926, 0.3420201]], "item_ids", &array![0u32, 1]);
        write_node(&store, "/lvl_2/node_1", &array![[1.0f32, 1.0], [0.5, 0.8660254]], "item_ids", &array![2u32, 3]);
        write_node(&store, "/lvl_2/node_2", &array![[0.0f32, 1.0], [-0.5, 0.8660254]], "item_ids", &array![4u32, 5]);
        write_node(&store, "/lvl_2/node_3", &array![[-0.8660254f32, 0.5], [-1.0, 0.0]], "item_ids", &array![6u32, 7]);

        let lvl_1 = vec![
            Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "node_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_1/node_1".to_string(), "node_ids".to_string()),
        ];
        let lvl_2 = vec![
            Node::new(as_readable_listable(&store), "/lvl_2/node_0".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_1".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_2".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_3".to_string(), "item_ids".to_string()),
        ];

        Index {
            metric: Metric::Cos,
            levels: 2,
            root: array![[1.0f32, 0.0], [1.0, 1.0]],
            nodes: vec![lvl_1, lvl_2],
            queries: Vec::new(),
        }
    }

    #[test]
    fn cos_search_returns_items_ordered_by_similarity_descending() {
        let mut index = build_cos_test_index();
        let query: Array1<f32> = array![1.0, 0.0];

        // search_exp=4 explores all 4 leaf nodes, so this is an exact top-8.
        let (items, _query_id) = index.new_search(query, 8, 4, -1, &HashSet::new());

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            "items should be ordered by descending cosine similarity to the query, most similar first"
        );

        // Cos should rank like IP (higher similarity = better), so the stored key
        // is the negated similarity - ascending key order means descending
        // similarity order, same convention as the IP metric.
        let expected_cos: [f32; 8] = [1.0, 0.9396926, 0.7071068, 0.5, 0.0, -0.5, -0.8660254, -1.0];
        let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
        for (score, expected) in scores.iter().zip(expected_cos.iter()) {
            assert!(
                (score - (-expected)).abs() < 1e-5,
                "score {score} should be -cos = {}",
                -expected
            );
        }
    }

    /// A 3-level tree: root -> lvl_1 -> lvl_2 -> lvl_3 (leaf). Every fixture so
    /// far tops out at levels=2, where the intermediate level (nodes[0]) always
    /// has `(level + 1) == (levels - 1)`, so its children are pushed straight as
    /// leaves (ecp_index.rs's `if` branch of that check). With levels=3, lvl_1's
    /// children (into lvl_2) instead take the `else` branch - pushed as another
    /// non-leaf level - which no existing test reaches. Only lvl_2's children
    /// (into lvl_3) hit the leaf branch.
    ///
    /// Kept deliberately linear (one child per intermediate node) so the descent
    /// path is unambiguous: root has 2 leaders, each lvl_1 node fans out to 2
    /// lvl_2 nodes, and each lvl_2 node has exactly 1 lvl_3 child - 4 items total,
    /// laid out on a line so nearest-to-farthest from the origin query is item
    /// id order, same shape of assertion as `l2_search_returns_nearest_items_in_order`.
    fn build_three_level_test_index() -> Index {
        let store = new_memory_store();

        write_node(&store, "/lvl_1/node_0", &array![[0.0f32, 0.0], [1.0, 1.0]], "node_ids", &array![0u32, 1]);
        write_node(&store, "/lvl_1/node_1", &array![[10.0f32, 10.0], [11.0, 11.0]], "node_ids", &array![2u32, 3]);

        write_node(&store, "/lvl_2/node_0", &array![[0.0f32, 0.0]], "node_ids", &array![0u32]);
        write_node(&store, "/lvl_2/node_1", &array![[1.0f32, 1.0]], "node_ids", &array![1u32]);
        write_node(&store, "/lvl_2/node_2", &array![[10.0f32, 10.0]], "node_ids", &array![2u32]);
        write_node(&store, "/lvl_2/node_3", &array![[11.0f32, 11.0]], "node_ids", &array![3u32]);

        write_node(&store, "/lvl_3/node_0", &array![[0.0f32, 0.0]], "item_ids", &array![0u32]);
        write_node(&store, "/lvl_3/node_1", &array![[1.0f32, 1.0]], "item_ids", &array![1u32]);
        write_node(&store, "/lvl_3/node_2", &array![[10.0f32, 10.0]], "item_ids", &array![2u32]);
        write_node(&store, "/lvl_3/node_3", &array![[11.0f32, 11.0]], "item_ids", &array![3u32]);

        let lvl_1 = vec![
            Node::new(as_readable_listable(&store), "/lvl_1/node_0".to_string(), "node_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_1/node_1".to_string(), "node_ids".to_string()),
        ];
        let lvl_2 = vec![
            Node::new(as_readable_listable(&store), "/lvl_2/node_0".to_string(), "node_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_1".to_string(), "node_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_2".to_string(), "node_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_2/node_3".to_string(), "node_ids".to_string()),
        ];
        let lvl_3 = vec![
            Node::new(as_readable_listable(&store), "/lvl_3/node_0".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_3/node_1".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_3/node_2".to_string(), "item_ids".to_string()),
            Node::new(as_readable_listable(&store), "/lvl_3/node_3".to_string(), "item_ids".to_string()),
        ];

        Index {
            metric: Metric::L2,
            levels: 3,
            root: array![[0.0f32, 0.0], [10.0, 10.0]],
            nodes: vec![lvl_1, lvl_2, lvl_3],
            queries: Vec::new(),
        }
    }

    #[test]
    fn three_level_tree_descends_through_intermediate_level() {
        let mut index = build_three_level_test_index();
        let query: Array1<f32> = array![0.0, 0.0];

        // search_exp=4 explores all 4 leaf nodes, so this is an exact top-4.
        let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);

        let scores: Vec<f32> = items.iter().map(|(d, _)| d.into_inner()).collect();
        assert!(scores.windows(2).all(|w| w[0] <= w[1]), "scores not sorted: {scores:?}");
    }

    #[test]
    fn levels_1_index_searches_like_ivf_without_panicking() {
        let mut index = build_ivf_style_index(Metric::L2);
        let query: Array1<f32> = array![0.0, 0.0];

        // In a levels=1 tree every popped node is a leaf, so leaf_cnt (what
        // search_exp actually counts) advances once per cluster regardless of
        // how many total node lookups that involves - search_exp=4 here means
        // "don't stop before all 4 clusters have been scanned", not "check 4
        // nodes" in general (those only coincide because there's nothing but
        // leaves in this particular tree).
        let (items, _query_id) = index.new_search(query, 4, 4, -1, &HashSet::new());

        let ids: Vec<u32> = items.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }
}
