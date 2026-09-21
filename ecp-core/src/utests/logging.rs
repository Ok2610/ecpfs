use super::*;

#[test]
fn format_entry_produces_valid_json_with_expected_fields() {
    let record = Record::builder()
        .level(log::Level::Debug)
        .target("ecp_core::build::builder")
        .args(format_args!("processing batch vecs 0..100"))
        .build();

    let line = format_entry(&record);
    let parsed: serde_json::Value =
        serde_json::from_str(&line).expect("format_entry must produce valid JSON");

    assert_eq!(parsed["level"], "DEBUG");
    assert_eq!(parsed["target"], "ecp_core::build::builder");
    assert_eq!(parsed["message"], "processing batch vecs 0..100");
    assert!(
        parsed["timestamp"].as_str().is_some(),
        "timestamp must be present"
    );
}

/// Sets and restores the process-wide max level immediately around the
/// check, since `init_writes_log_lines_to_its_file` also depends on it and
/// tests run in parallel.
#[test]
fn enabled_honors_the_configured_max_level() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let file = File::create(tmp.path().join("probe.jsonl")).expect("failed to create temp file");
    let logger = JsonlLogger {
        file: Mutex::new(file),
        write_failed: AtomicBool::new(false),
    };

    let original = log::max_level();
    log::set_max_level(LevelFilter::Warn);
    let warn_enabled = logger.enabled(&Metadata::builder().level(log::Level::Warn).build());
    let info_enabled = logger.enabled(&Metadata::builder().level(log::Level::Info).build());
    log::set_max_level(original);

    assert!(
        warn_enabled,
        "Warn must be enabled when the max level is Warn"
    );
    assert!(
        !info_enabled,
        "Info must not be enabled when the max level is Warn"
    );
}

#[test]
fn random_suffix_is_six_hex_chars() {
    let suffix = random_suffix();
    assert_eq!(suffix.len(), 6);
    assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
}

/// The only test in this crate that calls `init`, since `log`'s global logger
/// can only be set once per process. It looks for a unique marker rather than
/// exact file content, since other tests' log calls may land in the same file.
#[test]
fn init_writes_log_lines_to_its_file() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let path = init(Some(tmp.path()), LevelFilter::Debug).expect("failed to init logging");

    assert!(path.starts_with(tmp.path()));
    assert!(path.extension().is_some_and(|ext| ext == "jsonl"));

    log::debug!("unique-marker-for-init-test-12345");
    log::logger().flush();

    let contents = std::fs::read_to_string(&path).expect("failed to read log file");
    assert!(
        contents
            .lines()
            .any(|line| line.contains("unique-marker-for-init-test-12345")),
        "expected log file to contain the test's marker message, got:\n{contents}"
    );
}
