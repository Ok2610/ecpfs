use super::*;

use clap::error::ErrorKind;

/// Info is the new default, promoted from Debug.
#[test]
fn log_level_defaults_to_info() {
    let cli = Cli::try_parse_from(["ecp", "build-index", "embeddings.h5"]).unwrap();
    let Command::BuildIndex(args) = cli.command else {
        panic!("expected the build-index subcommand");
    };
    assert_eq!(args.logging.log_level, LogLevelArg::Info);
}

/// `info` didn't take `LoggingArgs` before, so a directory that isn't an
/// index couldn't report through the log like every other subcommand.
#[test]
fn info_subcommand_accepts_the_logging_flags() {
    let cli = Cli::try_parse_from([
        "ecp",
        "info",
        "my_index.zarr",
        "--with-logging",
        "--log-level",
        "debug",
    ])
    .unwrap();
    let Command::Info(args) = cli.command else {
        panic!("expected the info subcommand");
    };
    assert!(args.logging.with_logging);
    assert_eq!(args.logging.log_level, LogLevelArg::Debug);
}

/// A chunk size of 0 is rejected while parsing, on both chunk flags.
#[test]
fn a_chunk_size_of_zero_is_rejected_at_parsing() {
    for flag in ["--rep-chunk-mb", "--node-chunk-kb"] {
        let result = Cli::try_parse_from(["ecp", "build-index", "embeddings.h5", flag, "0"]);
        let Err(err) = result else {
            panic!("{flag} 0 was accepted");
        };
        assert_eq!(err.kind(), ErrorKind::ValueValidation, "{flag}: {err}");
    }
}

/// 1 is the smallest chunk size the parser accepts.
#[test]
fn a_chunk_size_of_one_is_accepted() {
    let cli = Cli::try_parse_from([
        "ecp",
        "build-index",
        "embeddings.h5",
        "--rep-chunk-mb",
        "1",
        "--node-chunk-kb",
        "1",
    ])
    .unwrap();
    let Command::BuildIndex(args) = cli.command else {
        panic!("expected the build-index subcommand");
    };
    assert_eq!(args.rep_chunk_mb, 1);
    assert_eq!(args.node_chunk_kb, 1);
}

/// The largest accepted chunk size still converts to bytes without overflow,
/// and one above it is rejected.
#[test]
fn chunk_sizes_are_limited_to_what_fits_in_bytes() {
    let rep_max = MAX_REP_CHUNK_MB.to_string();
    let node_max = MAX_NODE_CHUNK_KB.to_string();
    let cli = Cli::try_parse_from([
        "ecp",
        "build-index",
        "embeddings.h5",
        "--rep-chunk-mb",
        rep_max.as_str(),
        "--node-chunk-kb",
        node_max.as_str(),
    ])
    .unwrap();
    let Command::BuildIndex(args) = cli.command else {
        panic!("expected the build-index subcommand");
    };
    assert!(args.rep_chunk_mb.checked_mul(1024 * 1024).is_some());
    assert!(args.node_chunk_kb.checked_mul(1024).is_some());

    for (flag, too_big) in [
        ("--rep-chunk-mb", MAX_REP_CHUNK_MB + 1),
        ("--node-chunk-kb", MAX_NODE_CHUNK_KB + 1),
    ] {
        let result = Cli::try_parse_from([
            "ecp",
            "build-index",
            "embeddings.h5",
            flag,
            too_big.to_string().as_str(),
        ]);
        let Err(err) = result else {
            panic!("{flag} {too_big} was accepted");
        };
        assert_eq!(err.kind(), ErrorKind::ValueValidation, "{flag}: {err}");
    }
}
