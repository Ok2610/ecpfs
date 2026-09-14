use ordered_float::NotNan;
use std::cmp::Ordering;

use half::f16;
use ndarray::{Array1, Array2, Axis};
use zarrs::array::data_type::{float16, float32, int8, uint8};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

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

/// On-disk width for embeddings arrays. Narrower than `F32` means less disk
/// and less to read, but nothing is cached narrow: every read widens to f32
/// (see `read_subset_as_f32`), so resident size is the same whichever is
/// chosen. `F16` loses precision on genuinely `F32` data, while `UInt8`
/// (`0..=255`, SIFT-style descriptors) and `Int8` (`-128..=127`, symmetric
/// scalar quantization) round-trip integer-valued data exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDtype {
    UInt8,
    Int8,
    F16,
    F32,
}

impl EmbeddingDtype {
    /// True if storing as `self` cannot represent every value `native` can,
    /// so writing narrows the data. `Int8` and `UInt8` are each lossy for
    /// the other: neither range contains the other's.
    pub fn narrows(self, native: EmbeddingDtype) -> bool {
        use EmbeddingDtype::{F16, F32, Int8, UInt8};
        matches!(
            (self, native),
            (F16, F32)
                | (UInt8, F32)
                | (UInt8, F16)
                | (UInt8, Int8)
                | (Int8, F32)
                | (Int8, F16)
                | (Int8, UInt8)
        )
    }
}

/// The dtype `array`'s elements are stored as. `context` names the array in
/// the panic message when it holds a dtype ecpfs can't read.
pub fn dtype_of_array<T: ?Sized>(array: &Array<T>, context: &str) -> EmbeddingDtype {
    let dtype = array.data_type();
    if *dtype == float32() {
        EmbeddingDtype::F32
    } else if *dtype == float16() {
        EmbeddingDtype::F16
    } else if *dtype == uint8() {
        EmbeddingDtype::UInt8
    } else if *dtype == int8() {
        EmbeddingDtype::Int8
    } else {
        panic!(
            "unsupported embeddings dtype: {context} is {dtype:?} (use float32, float16, uint8 or int8)"
        )
    }
}

/// Reads `subset` of `array` as f32, widening from whatever dtype it's
/// stored as. Every distance computation runs on f32, so this is the one
/// place a stored dtype is widened on the read path.
pub fn read_subset_as_f32<T: ReadableStorageTraits + ?Sized + 'static>(
    array: &Array<T>,
    subset: &ArraySubset,
    context: &str,
) -> Array2<f32> {
    match dtype_of_array(array, context) {
        EmbeddingDtype::F32 => array
            .retrieve_array_subset::<Array2<f32>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}")),
        EmbeddingDtype::F16 => array
            .retrieve_array_subset::<Array2<f16>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}"))
            .mapv(|x| x.to_f32()),
        EmbeddingDtype::UInt8 => array
            .retrieve_array_subset::<Array2<u8>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}"))
            .mapv(|x| x as f32),
        EmbeddingDtype::Int8 => array
            .retrieve_array_subset::<Array2<i8>>(subset)
            .unwrap_or_else(|e| panic!("Failed to retrieve {context}: {e}"))
            .mapv(|x| x as f32),
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
            let q_2d = q.clone().insert_axis(Axis(0));
            let neg_dist_sq = negative_squared_distances(embeddings, &q_2d);
            neg_dist_sq.column(0).mapv(|v| (-v).sqrt())
        }
    }
}

/// `‖a−b‖² = ‖a‖² − 2·a·b + ‖b‖²`, negated so higher is always better -
/// shape `(a.nrows(), b.nrows())`. Shared by `calculate_distances` (search)
/// and `build::assign` (clustering), so the two never disagree on what L2
/// distance means.
pub fn negative_squared_distances(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let a_norms_sq = a.map_axis(Axis(1), |row| row.dot(&row));
    let b_norms_sq = b.map_axis(Axis(1), |row| row.dot(&row));
    let cross = a.dot(&b.t());

    let mut neg_dist_sq = 2.0 * cross;
    neg_dist_sq -= &a_norms_sq.insert_axis(Axis(1));
    neg_dist_sq -= &b_norms_sq.insert_axis(Axis(0));
    neg_dist_sq
}

/// A candidate node in a search's priority queue, ordered by `score`.
#[derive(Debug, Clone)]
pub struct HeapEntry {
    pub score: NotNan<f32>,
    pub is_leaf: i32,
    pub level: u32,
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

/// Fraction of total system RAM used as the default memory budget for
/// build and search when the caller doesn't specify one.
const DEFAULT_MEMORY_LIMIT_RAM_FRACTION: f64 = 0.8;

/// 80% of available RAM, floored to a whole gibibyte, in bytes. Uses the
/// enclosing cgroup's memory cap when one is set (Linux containers), since
/// `total_memory` otherwise reports host physical RAM regardless of it.
pub fn default_memory_limit_bytes() -> usize {
    let system = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    let total_ram_bytes = system
        .cgroup_limits()
        .map(|limits| limits.total_memory)
        .unwrap_or_else(|| system.total_memory());
    default_memory_limit_bytes_for(total_ram_bytes)
}

fn default_memory_limit_bytes_for(total_ram_bytes: u64) -> usize {
    const GIB: f64 = (1024 * 1024 * 1024) as f64;
    let total_gib = total_ram_bytes as f64 / GIB;
    let default_gib = (total_gib * DEFAULT_MEMORY_LIMIT_RAM_FRACTION).floor();
    (default_gib * GIB) as usize
}

#[cfg(test)]
#[path = "utests/utils.rs"]
mod tests;
