use super::*;

#[test]
fn format_entry_produces_valid_json_with_expected_fields() {
    let record = Record::builder()
        .level(log::Level::Debug)
        .target("ecp_core::build::builder")
        .args(format_args!("processing batch rows 0..100"))
        .build();

    let line = format_entry(&record);
    let parsed: serde_json::Value = serde_json::from_str(&line).expect("format_entry must produce valid JSON");

    assert_eq!(parsed["level"], "DEBUG");
    assert_eq!(parsed["target"], "ecp_core::build::builder");
    assert_eq!(parsed["message"], "processing batch rows 0..100");
    assert!(parsed["timestamp"].as_str().is_some(), "timestamp must be present");
}

#[test]
fn random_suffix_is_six_hex_chars() {
    let suffix = random_suffix();
    assert_eq!(suffix.len(), 6);
    assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
}

/// The only test in this crate allowed to call `init` for real - `log`'s
/// global logger can only be set once per process. Searches for a unique
/// marker rather than asserting exact file content, since other tests'
/// own log calls may land in the same file if they run concurrently.
#[test]
fn init_writes_real_log_lines_to_a_real_file() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let path = init(Some(tmp.path()), LevelFilter::Debug);

    assert!(path.starts_with(tmp.path()));
    assert!(path.extension().is_some_and(|ext| ext == "jsonl"));

    log::debug!("unique-marker-for-init-test-12345");
    log::logger().flush();

    let contents = std::fs::read_to_string(&path).expect("failed to read log file");
    assert!(
        contents.lines().any(|line| line.contains("unique-marker-for-init-test-12345")),
        "expected log file to contain the test's marker message, got:\n{contents}"
    );
}
