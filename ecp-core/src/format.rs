//! The on-disk index format's version, and the arrays a version 1 index holds.

/// The format version this build writes to `info/format_version` and reads.
pub const FORMAT_VERSION: u32 = 1;

/// The arrays every version 1 index holds besides `info/format_version`.
/// `info/is_normalized` is optional, so it is not listed.
pub(crate) const V1_ARRAYS: [&str; 6] = [
    "/info/levels",
    "/info/metric",
    "/info/total_items",
    "/info/next_item_id",
    "/rep_item_ids",
    "/index_root/embeddings",
];
