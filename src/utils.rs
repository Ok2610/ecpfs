use std::cmp::Ordering;
use ordered_float::NotNan;

use ndarray::{Array1, Array2, Axis};

pub enum Metric {
    L2,
    IP,
    Cos,
}

pub fn calculate_distances(
    embeddings: &Array2<f32>,
    q: &Array1<f32>,
    metric: &Metric,
) -> Array1<f32> { //(Vec<usize>, Array1<f32>) {
    assert_eq!(
        embeddings.ncols(),
        q.len(),
        "embeddings and query must have the same dim"
    );

    // compute the raw distance/similarity vector
    let distances: Array1<f32> = match metric {
        Metric::IP => {
            // matrix-vector multiply: all inner products
            embeddings.dot(q)
        }
        Metric::L2 => {
            // broadcast `q` to shape (n_embed, dim), subtract, then row-wise norm
            let diffs = embeddings - &q.broadcast((embeddings.nrows(), q.len())).unwrap();
            diffs
                .map_axis(Axis(1), |row| row.dot(&row).sqrt())
        }
        Metric::Cos => {
            // cosine = dot(e, q) / (‖e‖ · ‖q‖)
            let dots = embeddings.dot(q);
            let e_norms = embeddings
                .map_axis(Axis(1), |row| row.dot(&row).sqrt());
            let q_norm = q.dot(q).sqrt();
            &dots / &(e_norms * q_norm)
        }
    };


    // build a list of indices [0, 1, … n_embed-1]
    // let mut idx: Vec<usize> = (0..distances.len()).collect();

    // sort them by the metric
    // match metric {
    //     Metric::L2 => {
    //         // smaller distances first
    //         idx.sort_by(|&i, &j| {
    //             distances[i]
    //                 .partial_cmp(&distances[j])
    //                 .unwrap_or(Ordering::Equal)
    //         });
    //     }
    //     Metric::IP | Metric::Cos => {
    //         // larger similarities first
    //         idx.sort_by(|&i, &j| {
    //             distances[j]
    //                 .partial_cmp(&distances[i])
    //                 .unwrap_or(Ordering::Equal)
    //         });
    //     }
    // }

    // (idx, distances)
    distances
}

// pub trait AsF32 {
//     fn as_f32(self) -> f32;
// }

// impl AsF32 for f32 {
//     #[inline]
//     fn as_f32(self) -> f32 {
//         self
//     }
// }

// impl AsF32 for half::f16 {
//     #[inline]
//     fn as_f32(self) -> f32 {
//         self.to_f32()
//     }
// }



#[derive(Debug, Clone)]
pub struct HeapEntry {
    pub score: NotNan<f32>,
    pub is_leaf: i32,
    pub level:   u32,
    pub node_id: u32,
}

// We only compare on `score`:
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // forward to `Ord::cmp`
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare only on score:
        self.score.cmp(&other.score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::BinaryHeap;

    #[test]
    fn l2_distances_are_euclidean_norms() {
        let embeddings = array![[0.0f32, 0.0], [3.0, 4.0], [1.0, 0.0]];
        let q = array![0.0f32, 0.0];
        let distances = calculate_distances(&embeddings, &q, &Metric::L2);
        assert_eq!(distances.to_vec(), vec![0.0, 5.0, 1.0]);
    }

    #[test]
    fn ip_distances_are_dot_products() {
        let embeddings = array![[1.0f32, 0.0], [0.0, 1.0], [2.0, 3.0]];
        let q = array![2.0f32, 3.0];
        let distances = calculate_distances(&embeddings, &q, &Metric::IP);
        assert_eq!(distances.to_vec(), vec![2.0, 3.0, 13.0]);
    }

    #[test]
    fn cos_distances_are_bounded_by_one() {
        let embeddings = array![[1.0f32, 0.0], [1.0, 1.0], [-1.0, 0.0]];
        let q = array![1.0f32, 0.0];
        let distances = calculate_distances(&embeddings, &q, &Metric::Cos);
        assert!((distances[0] - 1.0).abs() < 1e-6);
        assert!((distances[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((distances[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "same dim")]
    fn calculate_distances_rejects_mismatched_dims() {
        let embeddings = array![[1.0f32, 2.0, 3.0]];
        let q = array![1.0f32, 2.0];
        let _ = calculate_distances(&embeddings, &q, &Metric::L2);
    }

    #[test]
    fn heap_entry_orders_by_score_only() {
        let mut heap = BinaryHeap::new();
        heap.push(HeapEntry {
            score: NotNan::new(1.0).unwrap(),
            is_leaf: 0,
            level: 5,
            node_id: 99,
        });
        heap.push(HeapEntry {
            score: NotNan::new(3.0).unwrap(),
            is_leaf: 1,
            level: 0,
            node_id: 1,
        });
        heap.push(HeapEntry {
            score: NotNan::new(2.0).unwrap(),
            is_leaf: 0,
            level: 2,
            node_id: 7,
        });

        // BinaryHeap is a max-heap: highest score pops first, regardless of the
        // other fields (level/node_id/is_leaf take no part in ordering).
        assert_eq!(heap.pop().unwrap().node_id, 1);
        assert_eq!(heap.pop().unwrap().node_id, 7);
        assert_eq!(heap.pop().unwrap().node_id, 99);
    }
}