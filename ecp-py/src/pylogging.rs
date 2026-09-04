use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use ecp_core::logging;

fn parse_level(level: &str) -> PyResult<log::LevelFilter> {
    level.parse().map_err(|_| {
        PyValueError::new_err(format!(
            "unknown log level {level:?} (use \"off\", \"trace\", \"debug\", \"info\", \"warn\" or \"error\")"
        ))
    })
}

/// init_logging(log_dir: Optional[str] = None, level: str = "debug") -> str
///
/// Starts file-based JSONL logging for this process (default `ecp_logs/`,
/// one file per process). Returns the resolved log file path. level="off"
/// still creates the file but logs nothing to it.
#[pyfunction]
#[pyo3(signature = (log_dir=None, level="debug"))]
pub fn init_logging(log_dir: Option<PathBuf>, level: &str) -> PyResult<String> {
    let level = parse_level(level)?;
    let path = logging::init(log_dir.as_deref(), level);
    Ok(path.display().to_string())
}
