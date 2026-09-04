use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use log::{LevelFilter, Log, Metadata, Record};
use rand::RngExt;
use time::format_description::well_known::Rfc3339;
use time::macros::format_description;
use time::OffsetDateTime;

static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

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

fn format_entry(record: &Record) -> String {
    let timestamp = OffsetDateTime::now_utc().format(&Rfc3339).unwrap_or_default();
    serde_json::json!({
        "timestamp": timestamp,
        "level": record.level().to_string(),
        "target": record.target(),
        "message": record.args().to_string(),
    })
    .to_string()
}

fn random_suffix() -> String {
    format!("{:06x}", rand::rng().random::<u32>() & 0xff_ffff)
}

/// Starts file-based JSONL logging for this process into `log_dir`
/// (default `ecp_logs/`), one file per process at `{timestamp}-{random}.jsonl`.
/// Idempotent: only the first call actually sets up logging (`log`'s global
/// logger can only be set once); later calls just return the same path.
pub fn init(log_dir: Option<&Path>, level: LevelFilter) -> PathBuf {
    LOG_PATH
        .get_or_init(|| {
            let dir = log_dir.map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("ecp_logs"));
            fs::create_dir_all(&dir).expect("Failed to create log directory");

            const TIMESTAMP_FORMAT: &[time::format_description::FormatItem] =
                format_description!("[year][month][day]T[hour][minute][second]Z");
            let timestamp =
                OffsetDateTime::now_utc().format(TIMESTAMP_FORMAT).expect("Failed to format timestamp");
            let path = dir.join(format!("{timestamp}-{}.jsonl", random_suffix()));

            let file = OpenOptions::new().create(true).append(true).open(&path).expect("Failed to open log file");
            if log::set_boxed_logger(Box::new(JsonlLogger { file: Mutex::new(file) })).is_ok() {
                log::set_max_level(level);
            }
            path
        })
        .clone()
}

#[cfg(test)]
#[path = "utests/logging.rs"]
mod tests;
