use super::*;
use ndarray::array;

#[test]
fn ip_assigns_each_data_point_to_the_highest_dot_product_representative() {
    // Counts are unequal so no index can coincidentally align between the two arrays.
    let representatives = array![[1.0f32, 0.0], [0.0, 1.0]];
    let data_points = array![[2.0f32, 0.0], [0.0, 3.0], [0.9, 1.1]];

    let (offsets, data) =
        determine_node_assignments(&representatives, &data_points, Metric::IP, false);

    assert_eq!(offsets.to_vec(), vec![0, 1, 3]);
    assert_eq!(data.to_vec(), vec![0, 1, 2]);
}

#[test]
fn l2_assigns_each_data_point_to_the_nearest_representative_by_distance() {
    // Counts are unequal so no index can coincidentally align between the two arrays.
    let representatives = array![[0.0f32, 0.0], [10.0, 10.0]];
    let data_points = array![[0.0f32, 1.0], [1.0, 0.0], [9.0, 9.0], [11.0, 11.0]];

    let (offsets, data) =
        determine_node_assignments(&representatives, &data_points, Metric::L2, false);

    // representative 0 gets data points 0,1; representative 1 gets 2,3
    assert_eq!(offsets.to_vec(), vec![0, 2, 4]);
    assert_eq!(data.to_vec(), vec![0, 1, 2, 3]);
}

/// On actual unit vectors, the `is_normalized` fast path must land on the
/// same assignment as the general path, not just avoid crashing.
#[test]
fn l2_with_is_normalized_true_matches_general_path_on_unit_vectors() {
    // Counts are unequal so no index can coincidentally align between the two arrays.
    let representatives = array![[1.0f32, 0.0], [0.0, 1.0]];
    let data_points = array![[1.0f32, 0.0], [0.6, 0.8], [0.0, 1.0]];

    let general = determine_node_assignments(&representatives, &data_points, Metric::L2, false);
    let fast_path = determine_node_assignments(&representatives, &data_points, Metric::L2, true);

    assert_eq!(general, fast_path);
}

/// Representatives 0 and 1 are identical so their scores tie exactly for
/// every data point; `max_by`'s tie-break (Iterator rule) keeps the higher
/// index.
#[test]
fn a_tied_score_is_broken_toward_the_higher_representative_index() {
    let representatives = array![[1.0f32, 0.0], [1.0, 0.0]];
    let data_points = array![[1.0f32, 0.0]];

    let (offsets, data) =
        determine_node_assignments(&representatives, &data_points, Metric::IP, false);

    assert_eq!(
        offsets.to_vec(),
        vec![0, 0, 1],
        "the tie must resolve to representative 1, not 0"
    );
    assert_eq!(data.to_vec(), vec![0]);
}

#[test]
fn group_by_assignments_preserves_order_within_each_group_and_handles_empty_groups() {
    // best_ids = [0, 2, 0, 2, 2], num_reps = 3.
    //   groups (intuitive): [[0, 2], [], [1, 3, 4]]
    //   offsets, data (returned): [0, 2, 2, 5], [0, 2, 1, 3, 4]
    let best_ids = array![0u32, 2, 0, 2, 2];

    let (offsets, data) = group_by_assignments(3, &best_ids);

    assert_eq!(
        offsets.to_vec(),
        vec![0, 2, 2, 5],
        "representative 1 gets no data points"
    );
    assert_eq!(data.to_vec(), vec![0, 2, 1, 3, 4]);
}
