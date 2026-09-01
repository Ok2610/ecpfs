use std::cmp::Ordering;
use ordered_float::NotNan;

use ndarray::{Array1, Array2, Axis};

/// `as_str`/`FromStr` round-trip through the strings stored in a built
/// index's `info/metric` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    L2,
    IP,
}

impl Metric {
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

/// `is_normalized`: true if every `embeddings` row is already unit-length,
/// letting `L2` skip computing their norms (`‖e−q‖² = ‖e‖² − 2·e·q + ‖q‖²`
/// with `‖e‖²` known to be 1). `q`'s norm is always computed, since a query
/// isn't guaranteed pre-normalized. No effect on `IP`.
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
        Metric::L2 if is_normalized => {
            let dots = embeddings.dot(q);
            let q_norm_sq = q.dot(q);
            (1.0 - 2.0 * dots + q_norm_sq).mapv(f32::sqrt)
        }
        Metric::L2 => {
            let diffs = embeddings - &q.broadcast((embeddings.nrows(), q.len())).unwrap();
            diffs.map_axis(Axis(1), |row| row.dot(&row).sqrt())
        }
    }
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
#[path = "utests/utils.rs"]
mod tests;