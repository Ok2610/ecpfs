use super::*;
use ndarray::array;
use std::str::FromStr;

#[test]
fn metric_as_str_and_from_str_round_trip() {
    for (metric, s) in [(Metric::L2, "L2"), (Metric::IP, "IP")] {
        assert_eq!(metric.as_str(), s);
        assert_eq!(Metric::from_str(s), Ok(metric));
    }
}

#[test]
fn metric_from_str_rejects_unknown_values() {
    assert!(Metric::from_str("l2").is_err(), "case must match exactly");
    assert!(Metric::from_str("euclidean").is_err());
}

#[test]
fn l2_distances_are_euclidean_norms() {
    let embeddings = array![[0.0f32, 0.0], [3.0, 4.0], [1.0, 0.0]];
    let q = array![0.0f32, 0.0];
    let distances = calculate_distances(&embeddings, &q, &Metric::L2, false);
    assert_eq!(distances.to_vec(), vec![0.0, 5.0, 1.0]);
}

/// Unit-length vectors with `is_normalized: true` take the shortcut formula,
/// and must give the same distances as the general formula, not just the
/// same ranking.
#[test]
fn l2_with_is_normalized_true_matches_general_formula_on_unit_vectors() {
    let embeddings = array![[0.0f32, 1.0], [0.6, 0.8], [1.0, 0.0]];
    let q = array![1.0f32, 0.0];

    let general = calculate_distances(&embeddings, &q, &Metric::L2, false);
    let fast_path = calculate_distances(&embeddings, &q, &Metric::L2, true);

    for (a, b) in general.iter().zip(fast_path.iter()) {
        assert!((a - b).abs() < 1e-6, "general={a}, fast_path={b}");
    }
}

/// Deterministic values of the magnitude an embedding has, at a dim wide
/// enough for float error to show.
fn pseudo_random_embeddings(rows: usize, dim: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, dim), |(r, c)| {
        let h = ((r * dim + c) as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    })
}

/// A row scored against itself cancels to near zero, and at a wide dim the
/// leftover error can come out positive, which used to make the root NaN and
/// panic the search. The dim-2 fixtures elsewhere are too narrow to reach it.
#[test]
fn l2_distance_from_a_vector_to_itself_is_zero_not_nan() {
    let embeddings = pseudo_random_embeddings(64, 1152);

    for row in 0..embeddings.nrows() {
        let q = embeddings.row(row).to_owned();
        let distances = calculate_distances(&embeddings, &q, &Metric::L2, false);
        assert!(
            distances.iter().all(|d| !d.is_nan()),
            "row {row} scored NaN against {:?}",
            distances.iter().position(|d| d.is_nan())
        );
        // The expanded form leaves an error of its own, so this lands near
        // zero rather than on it
        assert!(
            distances[row] < 0.05,
            "row {row} against itself scored {}",
            distances[row]
        );
    }
}

/// The same cancellation in the `is_normalized` shortcut.
#[test]
fn l2_normalized_distance_from_a_vector_to_itself_is_zero_not_nan() {
    let mut embeddings = pseudo_random_embeddings(64, 1152);
    for mut row in embeddings.rows_mut() {
        let norm = row.dot(&row).sqrt();
        row.mapv_inplace(|v| v / norm);
    }

    for row in 0..embeddings.nrows() {
        let q = embeddings.row(row).to_owned();
        let distances = calculate_distances(&embeddings, &q, &Metric::L2, true);
        assert!(
            distances.iter().all(|d| !d.is_nan()),
            "row {row} scored NaN"
        );
        assert!(
            distances[row] < 0.05,
            "row {row} against itself scored {}",
            distances[row]
        );
    }
}

#[test]
fn ip_distances_are_dot_products() {
    let embeddings = array![[1.0f32, 0.0], [0.0, 1.0], [2.0, 3.0]];
    let q = array![2.0f32, 3.0];
    let distances = calculate_distances(&embeddings, &q, &Metric::IP, false);
    assert_eq!(distances.to_vec(), vec![2.0, 3.0, 13.0]);
}

#[test]
#[should_panic(expected = "same dim")]
fn calculate_distances_rejects_mismatched_dims() {
    let embeddings = array![[1.0f32, 2.0, 3.0]];
    let q = array![1.0f32, 2.0];
    let _ = calculate_distances(&embeddings, &q, &Metric::L2, false);
}
