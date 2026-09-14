CLI (``ecp``)
=============

Installed alongside the Rust workspace as the ``ecp-cli`` crate (binary name
``ecp``). Build it with ``cargo build -p ecp-cli --release``.

.. code-block:: text

   Build and search eCP indexes

   Usage: ecp <COMMAND>

   Commands:
     build-index      Selects cluster representatives from `embeddings_file`, then builds the full tree over it into `save_file`
     add-data         Bulk-appends new vectors from `embeddings_file` into an already-built index
     search           Runs a single query, pulled from row `query_row` of `query_file`, against an existing index, or continues a previously-persisted one with `--resume`
     info             Prints an index's `info/*` metadata without loading its tree
     cleanup-queries  Erases persisted queries nobody has resumed, so `/queries/` doesn't grow forever on an index `search` keeps being run against
     help             Print this message or the help of the given subcommand(s)

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
             Width to write embeddings as. `native` matches the source; anything narrower than the source warns, since it loses precision and, for the integer dtypes, truncates fractions and clamps out-of-range values. Every read widens back to f32, so this saves disk, not memory [default: native] [possible values: native, uint8, int8, f16, f32]
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

add-data
--------

Offline counterpart to calling ``Index.insert`` from a live session. Loads
the index, bulk-appends every vector in ``embeddings_file``, exits.
New ids are assigned automatically from the index's ``next_item_id``, not
from ``total_items``, so ids left reserved by an interrupted earlier insert
are skipped rather than reused. The command prints the range it actually
assigned.

.. code-block:: bash

   ecp add-data my_index.zarr new_embeddings.h5

.. code-block:: text

   Usage: ecp add-data [OPTIONS] <INDEX_PATH> <EMBEDDINGS_FILE>

   Arguments:
     <INDEX_PATH>       Path to the index to insert into
     <EMBEDDINGS_FILE>  Zarr or HDF5 file with the new data vectors to append

   Options:
         --emb-grp-name <EMB_GRP_NAME>
             Group name for the embeddings dataset [default: embeddings]
         --fallback-batch-rows <FALLBACK_BATCH_ROWS>
             Row batch size used when the source has no natural on-disk chunk to align to (same meaning as build-index's flag of the same name) [default: 100000]
         --memory-limit-gb <MEMORY_LIMIT_GB>
             Caps how many touched nodes stay cached, in GB. Defaults to 80% of total system RAM
         --with-logging
             Turn on file-based logging for this run
         --log-dir <LOG_DIR>
             Directory to write the log file into
         --log-level <LOG_LEVEL>
             Log verbosity. `trace` also logs every node visited during search [default: debug] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help

search
------

Every call to ``search`` persists the query before it exits (a no-op if
nothing's left to resume once it finishes), printing ``query_id`` as its
first output line. Pass that id back via ``--resume`` in a later call to
continue the same query, without re-searching from the root:

.. code-block:: bash

   ecp search my_index.zarr query.h5 --k 5 --search-exp 1
   # query_id	0
   # 7	0.1234
   # 19	0.2201
   # ...

   ecp search my_index.zarr --resume 0 --k 5 --search-exp 1
   # query_id	0
   # 3	0.3350
   # ...

.. code-block:: text

   Usage: ecp search [OPTIONS] <INDEX_PATH> [QUERY_FILE]

   Arguments:
     <INDEX_PATH>  Path to the index to search
     [QUERY_FILE]  Zarr or HDF5 file to read the query vector from. Required unless --resume is given

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
         --resume <RESUME>
             Resume a previously-persisted query (its id is printed as this tool's first output line) instead of starting a new one. Ignores query_file/query_row/query_grp_name when given
         --with-logging
             Turn on file-based logging for this run
         --log-dir <LOG_DIR>
             Directory to write the log file into
         --log-level <LOG_LEVEL>
             Log verbosity. `trace` also logs every node visited during search [default: debug] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help

info
----

.. code-block:: text

   Usage: ecp info <INDEX_PATH>

   Arguments:
     <INDEX_PATH>  Path to the index to inspect

   Options:
     -h, --help  Print help

``Total Items`` counts the items actually stored, while ``Next Item Id`` is
the id the next insert will hand out. They match unless a crash mid-insert
left a reserved id range unwritten, in which case the id is ahead of the
count and ``info`` reports the size of the gap. Those ids are never reused.

cleanup-queries
----------------

Erases persisted queries older than ``--older-than-hours``, so an index that
``search`` keeps getting run against doesn't accumulate one ``/queries/{id}/``
group per invocation forever:

.. code-block:: bash

   # Erase anything persisted more than a day ago.
   ecp cleanup-queries my_index.zarr --older-than-hours 24

.. code-block:: text

   Usage: ecp cleanup-queries [OPTIONS] --older-than-hours <OLDER_THAN_HOURS> <INDEX_PATH>

   Arguments:
     <INDEX_PATH>  Path to the index to clean up

   Options:
         --older-than-hours <OLDER_THAN_HOURS>
             Erase any persisted query older than this many hours
         --with-logging
             Turn on file-based logging for this run
         --log-dir <LOG_DIR>
             Directory to write the log file into
         --log-level <LOG_LEVEL>
             Log verbosity. `trace` also logs every node visited during search [default: debug] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help
