use ndarray::{Array1, Array2, Axis};

use crate::utils::{negative_squared_distances, Metric};

/// For each vec of `data_embeddings`, finds the nearest vec of `node_embeddings`,
/// then groups the data vecs by which `node_embeddings` vec they landed on.
/// Returns `(offsets, data)`: node `n`'s assigned data vec indices are
/// `data[offsets[n]..offsets[n+1]]`.
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

/// Argmax of `matrix` along axis 0.
fn argmax_axis0(matrix: &Array2<f32>) -> Array1<u32> {
    matrix.map_axis(Axis(0), |scores| {
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("scores must not be NaN"))
            .map(|(node, _)| node as u32)
            .expect("scores must include at least one node")
    })
}

/// The equivalent of grouping data point indices into one list per
/// representative, just stored as two flat arrays instead of a list of
/// lists. Example: `best_ids = [1, 0, 1, 2]`, `num_reps = 3`.
///
///   groups (the intuitive shape): [[1], [0, 2], [3]]
///   offsets, data (what this returns): [0, 1, 3, 4], [1, 0, 2, 3]
///
/// `groups[r]` is `data[offsets[r]..offsets[r+1]]`.
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
