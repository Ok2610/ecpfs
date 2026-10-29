use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::metric::{Metric, negative_squared_distances};

/// Below this many rows, a batch is scored on the caller's thread.
/// Splitting it further would spend more on rayon's task overhead than
/// the extra core buys.
const MIN_ROWS_PER_CHUNK: usize = 256;

/// How many rows each parallel chunk gets: enough to spread `num_rows` over
/// every thread, never below `MIN_ROWS_PER_CHUNK`.
fn chunk_rows_for(num_rows: usize, num_threads: usize) -> usize {
    num_rows
        .div_ceil(num_threads.max(1))
        .max(MIN_ROWS_PER_CHUNK)
}

/// Decides which node each data vector belongs to. Finds the nearest row of
/// `node_embeddings` for each row of `data_embeddings`, then groups the rows by
/// node into `(offsets, data)`. Node `n`'s rows are `data[offsets[n]..offsets[n+1]]`.
pub fn determine_node_assignments(
    node_embeddings: &Array2<f32>,
    data_embeddings: &Array2<f32>,
    metric: Metric,
    is_normalized: bool,
) -> (Array1<u32>, Array1<u32>) {
    let best_ids = assign_to_nearest(node_embeddings, data_embeddings, metric, is_normalized);
    group_by_assignments(node_embeddings.nrows(), &best_ids)
}

/// Picks the nearest node for each data vector under `metric`. Returns one
/// `node_embeddings` row index for each row of `data_embeddings`, scored in
/// parallel row chunks of `data_embeddings`.
fn assign_to_nearest(
    node_embeddings: &Array2<f32>,
    data_embeddings: &Array2<f32>,
    metric: Metric,
    is_normalized: bool,
) -> Array1<u32> {
    let chunk_rows = chunk_rows_for(data_embeddings.nrows(), rayon::current_num_threads());
    score_in_chunks(
        node_embeddings,
        data_embeddings,
        metric,
        is_normalized,
        chunk_rows,
    )
}

/// Scores `data_embeddings` in row chunks of `chunk_rows`, one rayon task
/// per chunk, and concatenates each chunk's nearest-node ids back into a
/// single array in row order. Kept separate from `assign_to_nearest` so
/// tests can force more than one chunk regardless of the host's thread count.
fn score_in_chunks(
    node_embeddings: &Array2<f32>,
    data_embeddings: &Array2<f32>,
    metric: Metric,
    is_normalized: bool,
    chunk_rows: usize,
) -> Array1<u32> {
    let chunk_results: Vec<Array1<u32>> = data_embeddings
        .axis_chunks_iter(Axis(0), chunk_rows)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|chunk| score_chunk(node_embeddings, chunk, metric, is_normalized))
        .collect();

    Array1::from_iter(chunk_results.into_iter().flatten())
}

/// `assign_to_nearest` for one chunk: returns one `node_embeddings` row
/// index for each row of `data_chunk`.
fn score_chunk(
    node_embeddings: &Array2<f32>,
    data_chunk: ArrayView2<f32>,
    metric: Metric,
    is_normalized: bool,
) -> Array1<u32> {
    let scores: Array2<f32> = match metric {
        Metric::IP => node_embeddings.dot(&data_chunk.t()),
        // For unit vectors, nearest-by-L2 and highest-by-dot-product agree.
        Metric::L2 if is_normalized => node_embeddings.dot(&data_chunk.t()),
        Metric::L2 => negative_squared_distances(node_embeddings, &data_chunk.to_owned()),
    };
    argmax_axis0(&scores)
}

/// Argmax along axis 0: returns the highest-scoring row id for each column.
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

/// Groups data point indices by the representative they were assigned to,
/// stored as two flat arrays instead of a list of lists. Example:
/// `best_ids = [1, 0, 1, 2]`, `num_reps = 3`.
///
/// ```text
/// groups (the intuitive shape): [[1], [0, 2], [3]]
/// offsets, data (what this returns): [0, 1, 3, 4], [1, 0, 2, 3]
/// ```
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
