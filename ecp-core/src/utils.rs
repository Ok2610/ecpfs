//! The default memory limit. Also re-exports the `metric` and `dtype` items.

pub use crate::dtype::{EmbeddingDtype, dtype_of_array, read_subset_as_f32};
pub use crate::metric::{Metric, calculate_distances, negative_squared_distances};

/// Share of system RAM used as the memory limit when the caller doesn't set one.
const DEFAULT_MEMORY_LIMIT_RAM_FRACTION: f64 = 0.8;

/// Returns the default memory limit in bytes, 80% of RAM rounded down to a
/// whole GiB. Inside a Linux container it uses the container's memory cap,
/// since the host's RAM would be too high.
pub fn default_memory_limit_bytes() -> usize {
    let system = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    let total_ram_bytes = match system.cgroup_limits() {
        Some(limits) => limits.total_memory,
        None => {
            // No cgroup limits is the normal, every-call result outside
            // Linux, not a failure worth warning about there.
            if cfg!(target_os = "linux") {
                log::warn!(
                    "failed to read this Linux host's cgroup memory limit, \
                     falling back to its total RAM"
                );
            }
            system.total_memory()
        }
    };
    default_memory_limit_bytes_for(total_ram_bytes)
}

/// Computes the default memory limit for `total_ram_bytes` of RAM.
/// Note: This function is only split out from `default_memory_limit_bytes` for unit tests.
fn default_memory_limit_bytes_for(total_ram_bytes: u64) -> usize {
    const GIB: f64 = (1024 * 1024 * 1024) as f64;
    let total_gib = total_ram_bytes as f64 / GIB;
    let default_gib = (total_gib * DEFAULT_MEMORY_LIMIT_RAM_FRACTION).floor();
    (default_gib * GIB) as usize
}

#[cfg(test)]
#[path = "utests/utils.rs"]
mod tests;
