use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use ecp_core::build::builder::Builder;
use ecp_core::build::representatives::RepresentativeStrategy;
use ecp_core::build::source::EmbeddingsSource;
use ecp_core::search::Index;
use ecp_core::utils::Metric;

#[derive(Parser)]
#[command(name = "ecp", about = "Build and search eCP indexes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    BuildIndex(BuildIndexArgs),
    Search(SearchArgs),
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

/// Selects cluster representatives from `embeddings_file`, then builds the
/// full tree over it into `save_file`.
#[derive(clap::Args)]
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

    /// Group name for the embeddings dataset.
    #[arg(long, default_value = "embeddings")]
    emb_grp_name: String,

    /// How representatives are selected.
    #[arg(long, value_enum, default_value_t = RepSelectionArg::Offset)]
    rep_selection: RepSelectionArg,

    /// Memory budget for the build process, in GB (not strictly enforced).
    #[arg(long, default_value_t = 4)]
    memory_limit_gb: usize,

    /// Row batch size used when a source has no natural on-disk chunk to align to.
    #[arg(long, default_value_t = 100_000)]
    fallback_batch_rows: usize,
}

fn build_index(args: BuildIndexArgs) {
    let source = EmbeddingsSource::open(&args.embeddings_file, &args.emb_grp_name);
    let memory_limit_bytes = args.memory_limit_gb * 1024 * 1024 * 1024;
    let mut builder = Builder::create(&args.save_file, args.levels, args.metric.into(), args.is_normalized, memory_limit_bytes);
    builder.select_representatives(&source, args.target_cluster_items, args.rep_selection.into(), args.fallback_batch_rows);
    builder.build(&source, args.fallback_batch_rows);
}

/// Runs a single query, pulled from row `query_row` of `query_file`, against
/// an existing index.
#[derive(clap::Args)]
struct SearchArgs {
    /// Path to the index to search.
    index_path: PathBuf,

    /// Zarr or HDF5 file to read the query vector from.
    query_file: PathBuf,

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
}

fn search(args: SearchArgs) {
    let mut index = Index::load(args.index_path);
    let source = EmbeddingsSource::open(&args.query_file, &args.query_grp_name);
    let query = source.read_rows(args.query_row, args.query_row + 1).row(0).to_owned();
    let exclude = args.exclude.into_iter().collect();

    let (items, _query_id) = index.new_search(query, args.k, args.search_exp, args.max_increments, &exclude);
    for (distance, id) in items {
        println!("{id}\t{distance}");
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::BuildIndex(args) => build_index(args),
        Command::Search(args) => search(args),
    }
}
