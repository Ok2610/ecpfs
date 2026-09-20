use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use ecp_core::logging;

/// Parses a level name such as "debug", raising ValueError for an unknown one.
fn parse_level(level: &str) -> PyResult<log::LevelFilter> {
    level.parse().map_err(|_| {
        PyValueError::new_err(format!(
            "unknown log level {level:?} (use \"off\", \"trace\", \"debug\", \"info\", \"warn\" or \"error\")"
        ))
    })
}

/// init_logging(log_dir=None, level="debug")
///
/// Starts logging this process to a new JSONL file in ``log_dir`` (default
/// ``ecp_logs/``) and returns the file's path. ``level`` is "off", "trace",
/// "debug", "info", "warn" or "error"; "off" creates the file but logs nothing.
/// Only the first call sets logging up; later calls return the same path.
#[pyfunction]
#[pyo3(signature = (log_dir=None, level="debug"))]
pub fn init_logging(log_dir: Option<PathBuf>, level: &str) -> PyResult<String> {
    let level = parse_level(level)?;
    let path = logging::init(log_dir.as_deref(), level);
    Ok(path.display().to_string())
}
