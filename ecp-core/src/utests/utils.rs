use super::*;

#[test]
fn default_memory_limit_bytes_for_is_80_percent_of_total_ram_floored_to_a_gib() {
    const GIB: u64 = 1024 * 1024 * 1024;
    assert_eq!(default_memory_limit_bytes_for(4 * GIB), (3 * GIB) as usize);
    assert_eq!(default_memory_limit_bytes_for(8 * GIB), (6 * GIB) as usize);
    assert_eq!(
        default_memory_limit_bytes_for(16 * GIB),
        (12 * GIB) as usize
    );
}
