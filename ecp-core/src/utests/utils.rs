use super::*;
use ndarray::array;
use std::collections::BinaryHeap;
use std::str::FromStr;

#[test]
fn metric_as_str_and_from_str_round_trip() {
    for (metric, s) in [(Metric::L2, "L2"), (Metric::IP, "IP"), (Metric::Cos, "COS")] {
        assert_eq!(metric.as_str(), s);
        assert_eq!(Metric::from_str(s), Ok(metric));
    }
}

#[test]
fn metric_from_str_rejects_unknown_values() {
    assert!(Metric::from_str("l2").is_err(), "case must match exactly, like the PyO3 layer's existing parse_metric");
    assert!(Metric::from_str("euclidean").is_err());
}

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
