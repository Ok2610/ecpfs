use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand, ValueEnum};

use ecp_core::build::builder::Builder;
use ecp_core::build::builder::DEFAULT_MAX_CHUNK_BYTES;
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::logging;
use ecp_core::search::{Index, IndexInfo};
use ecp_core::utils::{EmbeddingDtype, Metric, default_memory_limit_bytes};

/// Default `--memory-limit-gb` for both subcommands: 80% of system RAM.
fn default_memory_limit_gib() -> usize {
    default_memory_limit_bytes() / (1024 * 1024 * 1024)
}

#[derive(Parser)]
#[command(name = "ecp", about = "Build and search eCP indexes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    BuildIndex(BuildIndexArgs),
    AddData(AddDataArgs),
    Search(SearchArgs),
    Info(InfoArgs),
    CleanupQueries(CleanupQueriesArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum MetricArg {
    L2,
    Ip,
}

impl From<MetricArg> for Metric {
    fn from(metric: MetricArg) -> Self {
        match metric {
            MetricArg::L2 => Metric::L2,
            MetricArg::Ip => Metric::IP,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum RepSelectionArg {
    Offset,
    Random,
}

impl From<RepSelectionArg> for RepresentativeStrategy {
    fn from(strategy: RepSelectionArg) -> Self {
        match strategy {
            RepSelectionArg::Offset => RepresentativeStrategy::Offset,
            RepSelectionArg::Random => RepresentativeStrategy::Random,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum EmbeddingDtypeArg {
    Native,
    // Without this, clap's kebab-casing renders the variant as `u-int8`.
    #[value(name = "uint8")]
    UInt8,
    Int8,
    F16,
    F32,
}

impl From<EmbeddingDtypeArg> for Option<EmbeddingDtype> {
    fn from(dtype: EmbeddingDtypeArg) -> Self {
        match dtype {
            EmbeddingDtypeArg::Native => None,
            EmbeddingDtypeArg::UInt8 => Some(EmbeddingDtype::UInt8),
            EmbeddingDtypeArg::Int8 => Some(EmbeddingDtype::Int8),
            EmbeddingDtypeArg::F16 => Some(EmbeddingDtype::F16),
            EmbeddingDtypeArg::F32 => Some(EmbeddingDtype::F32),
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum LogLevelArg {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<LogLevelArg> for log::LevelFilter {
    fn from(level: LogLevelArg) -> Self {
        match level {
            LogLevelArg::Off => log::LevelFilter::Off,
            LogLevelArg::Error => log::LevelFilter::Error,
            LogLevelArg::Warn => log::LevelFilter::Warn,
            LogLevelArg::Info => log::LevelFilter::Info,
            LogLevelArg::Debug => log::LevelFilter::Debug,
            LogLevelArg::Trace => log::LevelFilter::Trace,
        }
    }
}

/// Logging flags shared by every subcommand. Off by default; when
/// `with_logging` is set, writes JSONL to `log_dir` (default `ecp_logs/`).
#[derive(clap::Args)]
struct LoggingArgs {
    /// Turn on file-based logging for this run.
    #[arg(long)]
    with_logging: bool,

    /// Directory to write the log file into.
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Log verbosity. `trace` also logs every node visited during search.
    #[arg(long, value_enum, default_value_t = LogLevelArg::Debug)]
    log_level: LogLevelArg,
}

impl LoggingArgs {
    fn init_if_requested(&self) {
        if !self.with_logging {
            return;
        }
        let path = logging::init(self.log_dir.as_deref(), self.log_level.into());
        eprintln!("logging to {}", path.display());
    }
}

/// Selects cluster representatives from `embeddings_file`, then builds the
/// full tree over it into `save_file`.
#[derive(clap::Args)]
#[command(
    after_help = "Thread count is controlled by the RAYON_NUM_THREADS environment variable \
(e.g. RAYON_NUM_THREADS=4 ecp build-index ...), not a flag. It applies process-wide, for the \
lifetime of the run."
)]
struct BuildIndexArgs {
    /// Embeddings file with data vectors. Zarr or HDF5 file.
    embeddings_file: PathBuf,

    /// Output index path.
    #[arg(long, default_value = "ecpfs_index.zarr")]
    save_file: PathBuf,

    /// Levels in the index.
    #[arg(long, default_value_t = 3)]
    levels: u32,

    /// Preferred items for each cluster (no guarantees).
    #[arg(long, default_value_t = 100)]
    target_cluster_items: usize,

    /// Metric to use for distance calculations.
    #[arg(long, value_enum, default_value_t = MetricArg::L2)]
    metric: MetricArg,

    /// Set if every embedding is already unit-length, to skip norm computation.
    #[arg(long, default_value_t = false)]
    is_normalized: bool,

    /// Width to write embeddings as. `native` matches the source; anything
    /// narrower than the source warns, since it loses precision and, for
    /// the integer dtypes, truncates fractions and clamps out-of-range
    /// values. Every read widens back to f32, so this saves disk, not memory.
    #[arg(long, value_enum, default_value_t = EmbeddingDtypeArg::Native)]
    embedding_dtype: EmbeddingDtypeArg,

    /// Group name for the embeddings dataset.
    #[arg(long, default_value = "embeddings")]
    emb_grp_name: String,

    /// How representatives are selected.
    #[arg(long, value_enum, default_value_t = RepSelectionArg::Offset)]
    rep_selection: RepSelectionArg,

    /// Memory budget for the build process, in GB (not strictly enforced).
    /// Defaults to 80% of total system RAM.
    #[arg(long, default_value_t = default_memory_limit_gib())]
    memory_limit_gb: usize,

    /// Row batch size used when a source has no natural on-disk chunk to align to.
    #[arg(long, default_value_t = 100_000)]
    fallback_batch_rows: usize,

    /// Max size for one on-disk chunk, in MB.
    #[arg(long, default_value_t = DEFAULT_MAX_CHUNK_BYTES / (1024 * 1024))]
    max_chunk_mb: usize,

    #[command(flatten)]
    logging: LoggingArgs,
}

fn build_index(args: BuildIndexArgs) {
    args.logging.init_if_requested();
    let source = EmbeddingsSource::open(&args.embeddings_file, &args.emb_grp_name);
    let memory_limit_bytes = args.memory_limit_gb * 1024 * 1024 * 1024;
    let mut builder = Builder::create(
        &args.save_file,
        args.levels,
        args.metric.into(),
        args.is_normalized,
        memory_limit_bytes,
        args.embedding_dtype.into(),
        args.max_chunk_mb * 1024 * 1024,
    );
    builder.select_representatives(
        &source,
        args.target_cluster_items,
        args.rep_selection.into(),
        args.fallback_batch_rows,
    );
    builder.build(&source, args.fallback_batch_rows);
}

/// Bulk-appends new vectors from `embeddings_file` into an already-built
/// index. Offline counterpart to calling `Index::insert` from a live
/// session. One process, batches the file, exits.
#[derive(clap::Args)]
struct AddDataArgs {
    /// Path to the index to insert into.
    index_path: PathBuf,

    /// Zarr or HDF5 file with the new data vectors to append.
    embeddings_file: PathBuf,

    /// Group name for the embeddings dataset.
    #[arg(long, default_value = "embeddings")]
    emb_grp_name: String,

    /// Row batch size used when the source has no natural on-disk chunk to
    /// align to (same meaning as build-index's flag of the same name).
    #[arg(long, default_value_t = 100_000)]
    fallback_batch_rows: usize,

    /// Caps how many touched nodes stay cached, in GB. Defaults to 80% of
    /// total system RAM.
    #[arg(long, default_value_t = default_memory_limit_gib())]
    memory_limit_gb: usize,

    #[command(flatten)]
    logging: LoggingArgs,
}

fn add_data(args: AddDataArgs) {
    args.logging.init_if_requested();
    let source = EmbeddingsSource::open(&args.embeddings_file, &args.emb_grp_name);
    let memory_limit_bytes = args.memory_limit_gb * 1024 * 1024 * 1024;
    let index = Index::load(args.index_path, Some(memory_limit_bytes));

    let (total_vecs, _dim) = source.shape();
    let batch_vecs =
        source.chunk_aligned_batch_vecs(args.fallback_batch_rows, args.fallback_batch_rows);

    // Reported from what insert actually assigned, never predicted: a
    // reserved-but-lost range means ids are not simply "the old count onward".
    let mut first_id: Option<u32> = None;
    let mut last_id_end = 0u32;
    let mut start = 0usize;
    while start < total_vecs {
        let end = (start + batch_vecs).min(total_vecs);
        let assigned = index.insert(source.read_vecs(start, end));
        first_id.get_or_insert(assigned.start);
        last_id_end = assigned.end;
        start = end;
    }

    match first_id {
        Some(first) => println!("inserted {total_vecs} items (ids {first}..{last_id_end})"),
        None => println!("inserted 0 items"),
    }
}

/// Runs a query against an index, or continues a persisted one with
/// `--resume`. Persists before exiting; prints `query_id` as the first
/// output line.
#[derive(clap::Args)]
struct SearchArgs {
    /// Path to the index to search.
    index_path: PathBuf,

    /// Zarr or HDF5 file to read the query vector from. Required unless
    /// --resume is given.
    #[arg(required_unless_present = "resume")]
    query_file: Option<PathBuf>,

    /// Row within `query_file` to use as the query.
    #[arg(long, default_value_t = 0)]
    query_row: usize,

    /// Group name for the query dataset.
    #[arg(long, default_value = "embeddings")]
    query_grp_name: String,

    /// Number of items to return.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Search expansion factor.
    #[arg(long, default_value_t = 4)]
    search_exp: u32,

    /// Max retries when fewer than `k` items are found (-1 = unlimited).
    #[arg(long, default_value_t = -1)]
    max_increments: i32,

    /// Item ids to exclude, comma-separated.
    #[arg(long, value_delimiter = ',')]
    exclude: Vec<u32>,

    /// Caps how many touched nodes stay cached, in GB.
    /// Defaults to 80% of total system RAM.
    #[arg(long, default_value_t = default_memory_limit_gib())]
    memory_limit_gb: usize,

    /// Resume a persisted query (id printed as this tool's first output
    /// line) instead of starting a new one. Ignores
    /// query_file/query_row/query_grp_name.
    #[arg(long, conflicts_with = "query_file")]
    resume: Option<usize>,

    #[command(flatten)]
    logging: LoggingArgs,
}

fn search(args: SearchArgs) {
    args.logging.init_if_requested();
    let memory_limit_bytes = args.memory_limit_gb * 1024 * 1024 * 1024;
    let index = Index::load(args.index_path, Some(memory_limit_bytes));
    let exclude = args.exclude.into_iter().collect();

    let (items, query_id) = if let Some(query_id) = args.resume {
        let items = index.get_next_k_items(
            query_id,
            args.k,
            args.search_exp,
            args.max_increments,
            &exclude,
        );
        (items, query_id)
    } else {
        let query_file = args
            .query_file
            .expect("clap enforces this when --resume is absent");
        let source = EmbeddingsSource::open(&query_file, &args.query_grp_name);
        let query = source
            .read_vecs(args.query_row, args.query_row + 1)
            .row(0)
            .to_owned();
        index.new_search(
            query,
            args.k,
            args.search_exp,
            args.max_increments,
            &exclude,
        )
    };

    index.shutdown();

    println!("query_id\t{query_id}");
    for (distance, id) in items {
        println!("{id}\t{distance}");
    }
}

/// Prints an index's `info/*` metadata without loading its tree.
#[derive(clap::Args)]
struct InfoArgs {
    /// Path to the index to inspect.
    index_path: PathBuf,
}

fn info(args: InfoArgs) {
    let info = IndexInfo::load(args.index_path);
    println!("Levels: {}", info.levels);
    println!("Metric: {}", info.metric.as_str());
    println!("Normalized: {}", info.is_normalized);
    println!("Total Items: {}", info.total_items);
    println!("Next Item Id: {}", info.next_item_id);
    if info.next_item_id > info.total_items {
        println!(
            "  ({} id(s) reserved but never written, likely a crash mid-insert)",
            info.next_item_id - info.total_items
        );
    }
    println!("Total Representatives: {}", info.total_representatives);
}

/// Erases persisted queries nobody has resumed, so `/queries/` doesn't
/// grow forever on an index `search` keeps being run against.
#[derive(clap::Args)]
struct CleanupQueriesArgs {
    /// Path to the index to clean up.
    index_path: PathBuf,

    /// Erase any persisted query older than this many hours.
    #[arg(long)]
    older_than_hours: u64,

    #[command(flatten)]
    logging: LoggingArgs,
}

fn cleanup_queries(args: CleanupQueriesArgs) {
    args.logging.init_if_requested();
    let index = Index::load(args.index_path, None);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after 1970")
        .as_secs();
    let cutoff = now.saturating_sub(args.older_than_hours * 3600);

    let erased = index.cleanup_persisted_queries_older_than(cutoff);
    println!(
        "erased {erased} persisted quer{}",
        if erased == 1 { "y" } else { "ies" }
    );
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::BuildIndex(args) => build_index(args),
        Command::AddData(args) => add_data(args),
        Command::Search(args) => search(args),
        Command::Info(args) => info(args),
        Command::CleanupQueries(args) => cleanup_queries(args),
    }
}
