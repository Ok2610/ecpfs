//! Distance metrics and the distance computations built on them.

use ndarray::{Array1, Array2, Axis};

/// How the distance between two vectors is measured. `as_str` and `FromStr`
/// convert it to and from the name stored in an index's `info/metric`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance; lower is closer.
    L2,
    /// Inner product; higher is closer.
    IP,
}

impl Metric {
    /// Returns the metric's name, as stored in `info/metric`.
    pub fn as_str(self) -> &'static str {
        match self {
            Metric::L2 => "L2",
            Metric::IP => "IP",
        }
    }
}

impl std::str::FromStr for Metric {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "L2" => Ok(Metric::L2),
            "IP" => Ok(Metric::IP),
            other => Err(format!("unknown metric `{other}` (use \"L2\" or \"IP\")")),
        }
    }
}

/// Scores every row of `embeddings` against the query `q`, as L2 distances or,
/// for IP, inner products. Set `is_normalized` only if every row is
/// unit-length, which lets L2 skip computing the rows' norms.
/// L2 distance for unit-length rows = `sqrt(1 − 2·e·q + ‖q‖²)`.
pub fn calculate_distances(
    embeddings: &Array2<f32>,
    q: &Array1<f32>,
    metric: &Metric,
    is_normalized: bool,
) -> Array1<f32> {
    assert_eq!(
        embeddings.ncols(),
        q.len(),
        "embeddings and query must have the same dim"
    );

    match metric {
        Metric::IP => embeddings.dot(q),
        // Terms of similar size can cancel to a small negative where the true
        // distance is zero, and the root would turn that into NaN
        Metric::L2 if is_normalized => {
            let dots = embeddings.dot(q);
            let q_norm_sq = q.dot(q);
            (1.0 - 2.0 * dots + q_norm_sq).mapv(|v| v.max(0.0).sqrt())
        }
        Metric::L2 => {
            let q_2d = q.clone().insert_axis(Axis(0));
            let neg_dist_sq = negative_squared_distances(embeddings, &q_2d);
            neg_dist_sq.column(0).mapv(|v| (-v).max(0.0).sqrt())
        }
    }
}

/// Computes the squared L2 distance between every row of `a` and every row of
/// `b`, negated so a higher score is closer, as an `(a.nrows(), b.nrows())` matrix.
/// Squared distance = `‖a‖² − 2·a·b + ‖b‖²`.
pub fn negative_squared_distances(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let a_norms_sq = a.map_axis(Axis(1), |row| row.dot(&row));
    let b_norms_sq = b.map_axis(Axis(1), |row| row.dot(&row));
    let cross = a.dot(&b.t());

    let mut neg_dist_sq = 2.0 * cross;
    neg_dist_sq -= &a_norms_sq.insert_axis(Axis(1));
    neg_dist_sq -= &b_norms_sq.insert_axis(Axis(0));
    neg_dist_sq
}

#[cfg(test)]
#[path = "utests/metric.rs"]
mod tests;
