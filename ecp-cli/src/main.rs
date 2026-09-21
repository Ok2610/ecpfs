use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::builder::TypedValueParser;
use clap::{Parser, Subcommand, ValueEnum};

use ecp_core::build::builder::{
    Builder, ChunkSizes, DEFAULT_NODE_CHUNK_BYTES, DEFAULT_REP_CHUNK_BYTES,
};
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::logging;
use ecp_core::search::{Index, IndexInfo};
use ecp_core::utils::{EmbeddingDtype, Metric, default_memory_limit_bytes};

/// Largest --rep-chunk-mb whose size in bytes still fits in a `usize`.
const MAX_REP_CHUNK_MB: u64 = (usize::MAX / (1024 * 1024)) as u64;

/// Largest --node-chunk-kb whose size in bytes still fits in a `usize`.
const MAX_NODE_CHUNK_KB: u64 = (usize::MAX / 1024) as u64;

/// Returns the default --memory-limit-gb, 80% of RAM in whole GiB.
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
    /// Euclidean distance
    L2,
    /// Inner product
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
    /// Evenly spaced items
    Offset,
    /// A random sample
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
    /// The file's own type
    Native,
    /// 8-bit unsigned integers (0 to 255)
    // Without this, clap's kebab-casing renders the variant as `u-int8`.
    #[value(name = "uint8")]
    UInt8,
    /// 8-bit signed integers (-128 to 127)
    Int8,
    /// 16-bit floats
    F16,
    /// 32-bit floats
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

/// Logging flags shared by the subcommands. Logging is off unless
/// --with-logging is set.
#[derive(clap::Args)]
struct LoggingArgs {
    /// Log this run to a JSONL file.
    #[arg(long)]
    with_logging: bool,

    /// Directory for the log file, ecp_logs/ if not set.
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Log verbosity. trace also logs every node visited during search.
    #[arg(long, value_enum, default_value_t = LogLevelArg::Debug)]
    log_level: LogLevelArg,
}

impl LoggingArgs {
    /// Starts logging if --with-logging was given, and prints the log file's path.
    fn init_if_requested(&self) {
        if !self.with_logging {
            return;
        }
        let path = logging::init(self.log_dir.as_deref(), self.log_level.into());
        eprintln!("logging to {}", path.display());
    }
}

/// Builds a new index from EMBEDDINGS_FILE, saved at --save-file.
#[derive(clap::Args)]
#[command(
    after_help = "Thread count is controlled by the RAYON_NUM_THREADS environment variable \
(e.g. RAYON_NUM_THREADS=4 ecp build-index ...), not a flag. It applies process-wide, for the \
lifetime of the run."
)]
struct BuildIndexArgs {
    /// The embeddings to index, as a .zarr or .h5 file.
    embeddings_file: PathBuf,

    /// Where to save the index.
    #[arg(long, default_value = "ecpfs_index.zarr")]
    save_file: PathBuf,

    /// Number of node levels below the root.
    #[arg(long, default_value_t = 3)]
    levels: u32,

    /// Average number of items per cluster to aim for.
    #[arg(long, default_value_t = 100)]
    target_cluster_items: usize,

    /// How the distance between vectors is measured.
    #[arg(long, value_enum, default_value_t = MetricArg::L2)]
    metric: MetricArg,

    /// Set only if every embedding is unit-length. L2 then skips computing norms.
    #[arg(long, default_value_t = false)]
    is_normalized: bool,

    /// Type to store embeddings as on disk. A type narrower than the file's warns,
    /// since it loses precision, and integer types also drop fractions and clamp
    /// out-of-range values. Reads always widen to f32, so this saves disk, not memory.
    #[arg(long, value_enum, default_value_t = EmbeddingDtypeArg::Native)]
    embedding_dtype: EmbeddingDtypeArg,

    /// Name of the embeddings dataset inside the file.
    #[arg(long, default_value = "embeddings")]
    emb_grp_name: String,

    /// How to pick the representatives the tree is built from.
    #[arg(long, value_enum, default_value_t = RepSelectionArg::Offset)]
    rep_selection: RepSelectionArg,

    /// Target for the build's memory use in GB, not a hard cap. Defaults to 80%
    /// of RAM.
    #[arg(long, default_value_t = default_memory_limit_gib())]
    memory_limit_gb: usize,

    /// Chunk size assumed when the file isn't chunked, in rows.
    #[arg(long, default_value_t = 100_000)]
    fallback_batch_rows: usize,

    /// Chunk size for the representative arrays, in MB. Measure zarr read
    /// speed at a few chunk sizes on your own data before changing it.
    #[arg(
        long,
        default_value_t = DEFAULT_REP_CHUNK_BYTES / (1024 * 1024),
        value_parser = clap::value_parser!(u64).range(1..=MAX_REP_CHUNK_MB).map(|mb| mb as usize)
    )]
    rep_chunk_mb: usize,

    /// Chunk size for the tree nodes, in KB. Measure zarr read speed at a few
    /// chunk sizes on your own data before changing it.
    #[arg(
        long,
        default_value_t = DEFAULT_NODE_CHUNK_BYTES / 1024,
        value_parser = clap::value_parser!(u64).range(1..=MAX_NODE_CHUNK_KB).map(|kb| kb as usize)
    )]
    node_chunk_kb: usize,

    #[command(flatten)]
    logging: LoggingArgs,
}

/// Runs build-index, picking the representatives and then building the tree.
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
        ChunkSizes {
            rep_chunk_bytes: args.rep_chunk_mb * 1024 * 1024,
            node_chunk_bytes: args.node_chunk_kb * 1024,
        },
    );
    builder.select_representatives(
        &source,
        args.target_cluster_items,
        args.rep_selection.into(),
        args.fallback_batch_rows,
    );
    builder.build(&source, args.fallback_batch_rows);
}

/// Adds every vector in EMBEDDINGS_FILE to an existing index.
///
/// Prints the ids the new vectors got.
#[derive(clap::Args)]
struct AddDataArgs {
    /// The index to add to.
    index_path: PathBuf,

    /// The vectors to add, as a .zarr or .h5 file.
    embeddings_file: PathBuf,

    /// Name of the embeddings dataset inside the file.
    #[arg(long, default_value = "embeddings")]
    emb_grp_name: String,

    /// Rows added per batch, rounded up to whole chunks of the file.
    #[arg(long, default_value_t = 100_000)]
    fallback_batch_rows: usize,

    /// Cap on the index data kept in memory, in GB. Defaults to 80% of RAM.
    #[arg(long, default_value_t = default_memory_limit_gib())]
    memory_limit_gb: usize,

    #[command(flatten)]
    logging: LoggingArgs,
}

/// Runs add-data, inserting the file in batches and printing the ids assigned.
fn add_data(args: AddDataArgs) {
    args.logging.init_if_requested();
    let source = EmbeddingsSource::open(&args.embeddings_file, &args.emb_grp_name);
    let memory_limit_bytes = args.memory_limit_gb * 1024 * 1024 * 1024;
    let index = Index::load(args.index_path, Some(memory_limit_bytes));

    let (total_vecs, _dim) = source.shape();
    let batch_vecs =
        source.chunk_aligned_batch_vecs(args.fallback_batch_rows, args.fallback_batch_rows);

    // Report the ids insert returned. After a crashed insert, they don't simply
    // follow the old item count.
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

/// Searches an index for one query vector, or continues a saved query.
///
/// Prints the query id, then one line per result with the item id and score
/// separated by a tab, and saves the query so --resume can continue it.
#[derive(clap::Args)]
struct SearchArgs {
    /// The index to search.
    index_path: PathBuf,

    /// The .zarr or .h5 file holding the query vector. Required unless --resume
    /// is given.
    #[arg(required_unless_present = "resume")]
    query_file: Option<PathBuf>,

    /// Which row of QUERY_FILE is the query.
    #[arg(long, default_value_t = 0)]
    query_row: usize,

    /// Name of the query dataset inside the file.
    #[arg(long, default_value = "embeddings")]
    query_grp_name: String,

    /// Number of items to return.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// How many leaves to scan. More gives better results but a slower search.
    #[arg(long, default_value_t = 4)]
    search_exp: u32,

    /// How many times to double --search-exp while fewer than --k items are
    /// found. -1 for no limit.
    #[arg(long, default_value_t = -1)]
    max_increments: i32,

    /// Item ids to leave out, comma-separated.
    #[arg(long, value_delimiter = ',')]
    exclude: Vec<u32>,

    /// Cap on the index data kept in memory, in GB. Defaults to 80% of RAM.
    #[arg(long, default_value_t = default_memory_limit_gib())]
    memory_limit_gb: usize,

    /// Continue the saved query with this id, printed first by an earlier
    /// search, instead of starting a new one.
    #[arg(long, conflicts_with = "query_file")]
    resume: Option<usize>,

    #[command(flatten)]
    logging: LoggingArgs,
}

/// Runs search, printing the query id and then one line per result.
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

    // Save the query so --resume can continue it
    index.shutdown();

    println!("query_id\t{query_id}");
    for (score, id) in items {
        println!("{id}\t{score}");
    }
}

/// Prints an index's settings and item counts without loading its tree.
#[derive(clap::Args)]
struct InfoArgs {
    /// The index to describe.
    index_path: PathBuf,
}

/// Runs info, printing each setting on its own line.
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

/// Erases saved queries older than --older-than-hours.
///
/// Without it, repeated searches keep filling the index with saved queries.
#[derive(clap::Args)]
struct CleanupQueriesArgs {
    /// The index to clean up.
    index_path: PathBuf,

    /// Erase queries saved more than this many hours ago.
    #[arg(long)]
    older_than_hours: u64,

    #[command(flatten)]
    logging: LoggingArgs,
}

/// Runs cleanup-queries and prints how many queries it erased.
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

/// Parses the command line and runs the chosen subcommand.
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

#[cfg(test)]
#[path = "utests/main.rs"]
mod tests;
