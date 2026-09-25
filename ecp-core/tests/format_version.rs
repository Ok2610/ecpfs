//! Tests the stored format version against indexes built and saved on disk.

mod common;

use std::path::Path;
use std::sync::Arc;

use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;

use common::{build_index, two_clusters};
use ecp_core::EcpError;
use ecp_core::format::FORMAT_VERSION;
use ecp_core::search::{Index, IndexInfo};

/// Overwrites the `/info/{name}` scalar of a built index.
fn overwrite_info_u32(index_path: &Path, name: &str, value: u32) {
    let store: zarrs::storage::ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(index_path).expect("failed to reopen store"));
    let array =
        Array::open(store, &format!("/info/{name}")).expect("failed to open the info field");
    array
        .store_chunk(&[], vec![value])
        .expect("failed to overwrite the info field");
}

/// Deletes `info/format_version`, leaving an index as an earlier build wrote it.
fn remove_format_version(index_path: &Path) {
    std::fs::remove_dir_all(index_path.join("info/format_version"))
        .expect("failed to remove info/format_version");
}

#[test]
fn a_built_index_carries_the_current_format_version() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);

    let info = IndexInfo::load(index_path).unwrap();

    assert_eq!(info.format_version, FORMAT_VERSION);
}

#[test]
fn index_load_adds_a_deleted_format_version_back_and_index_info_does_not() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);
    remove_format_version(&index_path);

    let info = IndexInfo::load(index_path.clone()).unwrap();
    assert_eq!(info.format_version, FORMAT_VERSION);
    assert!(
        !index_path.join("info/format_version").exists(),
        "IndexInfo is read-only and must not write the field"
    );

    Index::load(index_path.clone(), None).unwrap();
    assert!(index_path.join("info/format_version/zarr.json").exists());
}

/// A read-only copy of an old index cannot take the write, and must still open.
#[cfg(unix)]
#[test]
fn a_read_only_index_without_format_version_still_loads() {
    use std::os::unix::fs::PermissionsExt;

    let (_tmp, index_path) = build_index(&two_clusters(), None);
    remove_format_version(&index_path);
    let info_dir = index_path.join("info");
    std::fs::set_permissions(&info_dir, std::fs::Permissions::from_mode(0o555)).unwrap();

    // A superuser writes through a read-only directory, which voids the test
    let probe = info_dir.join("probe");
    let is_superuser = std::fs::File::create(&probe).is_ok();
    let loaded = Index::load(index_path.clone(), None);
    let added = index_path.join("info/format_version").exists();
    std::fs::set_permissions(&info_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    if is_superuser {
        return;
    }

    assert!(loaded.is_ok(), "a read-only index must still load");
    assert!(!added);
}

#[test]
fn a_format_version_from_a_newer_release_is_refused_by_both_loaders() {
    let (_tmp, index_path) = build_index(&two_clusters(), None);
    overwrite_info_u32(&index_path, "format_version", FORMAT_VERSION + 1);

    let info = IndexInfo::load(index_path.clone());
    let index = Index::load(index_path, None);

    assert!(matches!(info, Err(EcpError::UnsupportedVersion(_))));
    assert!(matches!(index, Err(EcpError::UnsupportedVersion(_))));
}
