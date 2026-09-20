use super::*;

/// Int8 and UInt8 each narrow the other, since neither range contains the
/// other's.
#[test]
fn narrows_flags_every_lossy_direction_and_no_lossless_one() {
    use EmbeddingDtype::{F16, F32, Int8, UInt8};

    for (target, native) in [
        (F16, F32),
        (UInt8, F32),
        (UInt8, F16),
        (UInt8, Int8),
        (Int8, F32),
        (Int8, F16),
        (Int8, UInt8),
    ] {
        assert!(
            target.narrows(native),
            "storing {native:?} data as {target:?} loses values"
        );
    }

    for (target, native) in [
        (F32, F16),
        (F32, UInt8),
        (F32, Int8),
        (F16, UInt8),
        (F16, Int8),
        (F32, F32),
        (F16, F16),
        (UInt8, UInt8),
        (Int8, Int8),
    ] {
        assert!(
            !target.narrows(native),
            "storing {native:?} data as {target:?} is lossless, so it must not warn"
        );
    }
}
