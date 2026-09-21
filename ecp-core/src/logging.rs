//! Optional logging to a JSONL file, one JSON object per line.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use log::{LevelFilter, Log, Metadata, Record};
use rand::RngExt;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use time::macros::format_description;

use crate::error::{Result, ResultExt};

/// The log file's path, or the failure that stopped `init` from setting one
/// up. Only the first call runs; a failing first call is cached too, so a
/// later call with a different `log_dir` does not get a second attempt.
static LOG_PATH: OnceLock<Result<PathBuf>> = OnceLock::new();

/// Writes each log record as one JSON line to `file`.
struct JsonlLogger {
    file: Mutex<File>,
}

impl Log for JsonlLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        let entry = format_entry(record);
        if let Ok(mut file) = self.file.lock() {
            let _ = writeln!(file, "{entry}");
        }
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

/// Formats `record` as one JSON object with its timestamp, level, target and
/// message.
fn format_entry(record: &Record) -> String {
    let timestamp = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_default();
    serde_json::json!({
        "timestamp": timestamp,
        "level": record.level().to_string(),
        "target": record.target(),
        "message": record.args().to_string(),
    })
    .to_string()
}

/// Returns 6 random hex digits, so two processes started in the same second
/// get different log files.
fn random_suffix() -> String {
    format!("{:06x}", rand::rng().random::<u32>() & 0xff_ffff)
}

/// Starts logging this process to a new JSONL file in `log_dir` (default
/// `ecp_logs/`), keeping records at `level` and above, and returns its path.
/// Only the first call sets logging up; later calls return the same path.
pub fn init(log_dir: Option<&Path>, level: LevelFilter) -> Result<PathBuf> {
    LOG_PATH
        .get_or_init(|| {
            let dir = log_dir
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("ecp_logs"));
            fs::create_dir_all(&dir).store_err("failed to create log directory")?;

            // One file per process, named by its start time plus a random suffix
            const TIMESTAMP_FORMAT: &[time::format_description::FormatItem] =
                format_description!("[year][month][day]T[hour][minute][second]Z");
            let timestamp = OffsetDateTime::now_utc()
                .format(TIMESTAMP_FORMAT)
                .store_err("failed to format the log file's timestamp")?;
            let path = dir.join(format!("{timestamp}-{}.jsonl", random_suffix()));

            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .store_err("failed to open log file")?;
            // Leave the level alone if another logger was set first
            if log::set_boxed_logger(Box::new(JsonlLogger {
                file: Mutex::new(file),
            }))
            .is_ok()
            {
                log::set_max_level(level);
            }
            Ok(path)
        })
        .clone()
}

#[cfg(test)]
#[path = "utests/logging.rs"]
mod tests;
