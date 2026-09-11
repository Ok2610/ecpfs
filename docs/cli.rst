CLI (``ecp``)
=============

Installed alongside the Rust workspace as the ``ecp-cli`` crate (binary name
``ecp``). Build it with ``cargo build -p ecp-cli --release``.

.. code-block:: text

   Build and search eCP indexes

   Usage: ecp <COMMAND>

   Commands:
     build-index  Selects cluster representatives from `embeddings_file`, then builds the full tree over it into `save_file`
     search       Runs a single query, pulled from row `query_row` of `query_file`, against an existing index
     help         Print this message or the help of the given subcommand(s)

build-index
-----------

.. code-block:: text

   Usage: ecp build-index [OPTIONS] <EMBEDDINGS_FILE>

   Arguments:
     <EMBEDDINGS_FILE>  Embeddings file with data vectors. Zarr or HDF5 file

   Options:
         --save-file <SAVE_FILE>
             Output index path [default: ecpfs_index.zarr]
         --levels <LEVELS>
             Levels in the index [default: 3]
         --target-cluster-items <TARGET_CLUSTER_ITEMS>
             Preferred items for each cluster (no guarantees) [default: 100]
         --metric <METRIC>
             Metric to use for distance calculations [default: l2] [possible values: l2, ip]
         --is-normalized
             Set if every embedding is already unit-length, to skip norm computation
         --embedding-dtype <EMBEDDING_DTYPE>
             Precision to write embeddings as. `native` matches the source (warns if `f16` is forced against an `f32` source, a real precision loss) [default: native] [possible values: native, f16, f32]
         --emb-grp-name <EMB_GRP_NAME>
             Group name for the embeddings dataset [default: embeddings]
         --rep-selection <REP_SELECTION>
             How representatives are selected [default: offset] [possible values: offset, random]
         --memory-limit-gb <MEMORY_LIMIT_GB>
             Memory budget for the build process, in GB (not strictly enforced). Defaults to 80% of total system RAM
         --fallback-batch-rows <FALLBACK_BATCH_ROWS>
             Row batch size used when a source has no natural on-disk chunk to align to [default: 100000]
         --max-chunk-mb <MAX_CHUNK_MB>
             Max size for one on-disk chunk, in MB [default: 50]
         --with-logging
             Turn on file-based logging for this run
         --log-dir <LOG_DIR>
             Directory to write the log file into
         --log-level <LOG_LEVEL>
             Log verbosity. `trace` also logs every node visited during search [default: debug] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help

   Thread count is controlled by the RAYON_NUM_THREADS environment variable
   (e.g. RAYON_NUM_THREADS=4 ecp build-index ...), not a flag - it applies
   process-wide, for the lifetime of the run.

search
------

.. code-block:: text

   Usage: ecp search [OPTIONS] <INDEX_PATH> <QUERY_FILE>

   Arguments:
     <INDEX_PATH>  Path to the index to search
     <QUERY_FILE>  Zarr or HDF5 file to read the query vector from

   Options:
         --query-row <QUERY_ROW>
             Row within `query_file` to use as the query [default: 0]
         --query-grp-name <QUERY_GRP_NAME>
             Group name for the query dataset [default: embeddings]
         --k <K>
             Number of items to return [default: 10]
         --search-exp <SEARCH_EXP>
             Search expansion factor [default: 4]
         --max-increments <MAX_INCREMENTS>
             Max retries when fewer than `k` items are found (-1 = unlimited) [default: -1]
         --exclude <EXCLUDE>
             Item ids to exclude, comma-separated
         --memory-limit-gb <MEMORY_LIMIT_GB>
             Caps how many touched nodes stay cached (LRU-evicted), in GB. Defaults to 80% of total system RAM
         --with-logging
             Turn on file-based logging for this run
         --log-dir <LOG_DIR>
             Directory to write the log file into
         --log-level <LOG_LEVEL>
             Log verbosity. `trace` also logs every node visited during search [default: debug] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help
