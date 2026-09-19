pub use crate::dtype::{EmbeddingDtype, dtype_of_array, read_subset_as_f32};
pub use crate::metric::{Metric, calculate_distances, negative_squared_distances};

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
