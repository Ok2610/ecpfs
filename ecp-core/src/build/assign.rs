use ndarray::{Array1, Array2, Axis};

use crate::utils::{negative_squared_distances, Metric};

/// Determines which representative each data point should be clustered under.
pub fn determine_node_assignments(
    node_embeddings: &Array2<f32>,
    data_embeddings: &Array2<f32>,
    metric: Metric,
    is_normalized: bool,
) -> (Array1<u32>, Array1<u32>) {
    let best_ids = assign_to_nearest(node_embeddings, data_embeddings, metric, is_normalized);
    group_by_assignments(node_embeddings.nrows(), &best_ids)
}

/// Picks the nearest representative for each data point under `metric`.
fn assign_to_nearest(
    node_embeddings: &Array2<f32>,
    data_embeddings: &Array2<f32>,
    metric: Metric,
    is_normalized: bool,
) -> Array1<u32> {
    let scores: Array2<f32> = match metric {
        Metric::IP => node_embeddings.dot(&data_embeddings.t()),
        // For unit vectors, nearest-by-L2 and highest-by-dot-product agree.
        Metric::L2 if is_normalized => node_embeddings.dot(&data_embeddings.t()),
        Metric::L2 => negative_squared_distances(node_embeddings, data_embeddings),
    };
    argmax_axis0(&scores)
}

/// Picks the winning representative for each data point (column) of a score matrix.
fn argmax_axis0(matrix: &Array2<f32>) -> Array1<u32> {
    Array1::from_iter(matrix.axis_iter(Axis(1)).map(|column| {
        column
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("scores must not be NaN"))
            .map(|(representative, _)| representative as u32)
            .expect("node_embeddings must have at least one representative")
    }))
}

/// Turns per-data-point assignments into per-representative groups, so the
/// tree builder can process one representative's data points at a time.
pub fn group_by_assignments(num_reps: usize, best_ids: &Array1<u32>) -> (Array1<u32>, Array1<u32>) {
    let mut counts = vec![0u32; num_reps];
    for &rep in best_ids.iter() {
        counts[rep as usize] += 1;
    }

    let mut offsets = Array1::<u32>::zeros(num_reps + 1);
    for r in 0..num_reps {
        offsets[r + 1] = offsets[r] + counts[r];
    }

    let mut cursor: Vec<u32> = offsets.iter().take(num_reps).copied().collect();
    let mut data = Array1::<u32>::zeros(best_ids.len());
    for (data_point, &rep) in best_ids.iter().enumerate() {
        let pos = cursor[rep as usize];
        data[pos as usize] = data_point as u32;
        cursor[rep as usize] += 1;
    }

    (offsets, data)
}

#[cfg(test)]
#[path = "utests/assign.rs"]
mod tests;
