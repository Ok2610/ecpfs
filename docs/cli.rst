CLI (``ecp``)
=============

``ecp`` is the command line tool, built from the ``ecp-cli`` crate with
``cargo build -p ecp-cli --release``. The option lists below are its ``-h``
output; ``--help`` adds a longer description of each choice.

.. code-block:: text

   Build and search eCP indexes

   Usage: ecp <COMMAND>

   Commands:
     build-index      Builds a new index from EMBEDDINGS_FILE, saved at --save-file
     add-data         Adds every vector in EMBEDDINGS_FILE to an existing index
     search           Searches an index for one query vector, or continues a saved query
     info             Prints an index's settings and item counts without loading its tree
     cleanup-queries  Erases saved queries older than --older-than-hours
     help             Print this message or the help of the given subcommand(s)

   Options:
     -h, --help  Print help

build-index
-----------

.. code-block:: bash

   ecp build-index embeddings.h5 --save-file my_index.zarr --levels 3

.. code-block:: text

   Usage: ecp build-index [OPTIONS] <EMBEDDINGS_FILE>

   Arguments:
     <EMBEDDINGS_FILE>  The embeddings to index, as a .zarr or .h5 file

   Options:
         --save-file <SAVE_FILE>
             Where to save the index [default: ecpfs_index.zarr]
         --levels <LEVELS>
             Number of node levels below the root [default: 3]
         --target-cluster-items <TARGET_CLUSTER_ITEMS>
             Average number of items per cluster to aim for [default: 100]
         --metric <METRIC>
             How the distance between vectors is measured [default: l2] [possible values: l2, ip]
         --is-normalized
             Set only if every embedding is unit-length. L2 then skips computing norms
         --embedding-dtype <EMBEDDING_DTYPE>
             Type to store embeddings as on disk. A type narrower than the file's warns, since it loses precision, and integer types also drop fractions and clamp out-of-range values. Reads always widen to f32, so this saves disk, not memory [default: native] [possible values: native, uint8, int8, f16, f32]
         --emb-grp-name <EMB_GRP_NAME>
             Name of the embeddings dataset inside the file [default: embeddings]
         --rep-selection <REP_SELECTION>
             How to pick the representatives the tree is built from [default: offset] [possible values: offset, random]
         --memory-limit-gb <MEMORY_LIMIT_GB>
             Target for the build's memory use in GB, not a hard cap. Defaults to 80% of RAM
         --fallback-batch-rows <FALLBACK_BATCH_ROWS>
             Chunk size assumed when the file isn't chunked, in rows [default: 100000]
         --rep-chunk-mb <REP_CHUNK_MB>
             Chunk size for the representative arrays, in MB. Measure zarr read speed at a few chunk sizes on your own data before changing it [default: 8]
         --node-chunk-kb <NODE_CHUNK_KB>
             Chunk size for the tree nodes, in KB. Measure zarr read speed at a few chunk sizes on your own data before changing it [default: 512]
         --with-logging
             Log this run to a JSONL file
         --log-dir <LOG_DIR>
             Directory for the log file, ecp_logs/ if not set
         --log-level <LOG_LEVEL>
             Log verbosity. trace also logs every node visited during search [default: info] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help (see more with '--help')

   Thread count is controlled by the RAYON_NUM_THREADS environment variable (e.g. RAYON_NUM_THREADS=4 ecp build-index ...), not a flag. It applies process-wide, for the lifetime of the run.

add-data
--------

Adds every vector in a file to an existing index, the command line version of
``Index.insert``. New ids start at the index's ``next_item_id``, so ids left
unused by a crashed earlier insert are skipped, never reused. The command
prints the ids it assigned.

.. code-block:: bash

   ecp add-data my_index.zarr new_embeddings.h5

.. code-block:: text

   Usage: ecp add-data [OPTIONS] <INDEX_PATH> <EMBEDDINGS_FILE>

   Arguments:
     <INDEX_PATH>       The index to add to
     <EMBEDDINGS_FILE>  The vectors to add, as a .zarr or .h5 file

   Options:
         --emb-grp-name <EMB_GRP_NAME>
             Name of the embeddings dataset inside the file [default: embeddings]
         --fallback-batch-rows <FALLBACK_BATCH_ROWS>
             Rows added per batch, rounded up to whole chunks of the file [default: 100000]
         --memory-limit-gb <MEMORY_LIMIT_GB>
             Cap on the index data kept in memory, in GB. Defaults to 80% of RAM
         --with-logging
             Log this run to a JSONL file
         --log-dir <LOG_DIR>
             Directory for the log file, ecp_logs/ if not set
         --log-level <LOG_LEVEL>
             Log verbosity. trace also logs every node visited during search [default: info] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help (see more with '--help')

search
------

Every ``search`` saves its query before exiting, unless nothing is left to
return, and prints the query id first. Pass that id to ``--resume`` later to
continue the same query without starting over. :doc:`search-parameters` explains ``--k``,
``--search-exp``, ``--max-increments`` and ``--exclude``.

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
     <INDEX_PATH>  The index to search
     [QUERY_FILE]  The .zarr or .h5 file holding the query vector. Required unless --resume is given

   Options:
         --query-row <QUERY_ROW>
             Which row of QUERY_FILE is the query [default: 0]
         --query-grp-name <QUERY_GRP_NAME>
             Name of the query dataset inside the file [default: embeddings]
         --k <K>
             Number of items to return [default: 10]
         --search-exp <SEARCH_EXP>
             How many leaves to scan. More gives better results but a slower search [default: 4]
         --max-increments <MAX_INCREMENTS>
             How many times to double --search-exp while fewer than --k items are found. -1 for no limit [default: -1]
         --exclude <EXCLUDE>
             Item ids to leave out, comma-separated
         --memory-limit-gb <MEMORY_LIMIT_GB>
             Cap on the index data kept in memory, in GB. Defaults to 80% of RAM
         --resume <RESUME>
             Continue the saved query with this id, printed first by an earlier search, instead of starting a new one
         --with-logging
             Log this run to a JSONL file
         --log-dir <LOG_DIR>
             Directory for the log file, ecp_logs/ if not set
         --log-level <LOG_LEVEL>
             Log verbosity. trace also logs every node visited during search [default: info] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help (see more with '--help')

info
----

.. code-block:: text

   Usage: ecp info [OPTIONS] <INDEX_PATH>

   Arguments:
     <INDEX_PATH>  The index to describe

   Options:
         --with-logging           Log this run to a JSONL file
         --log-dir <LOG_DIR>      Directory for the log file, ecp_logs/ if not set
         --log-level <LOG_LEVEL>  Log verbosity. trace also logs every node visited during search [default: info] [possible values: off, error, warn, info, debug, trace]
     -h, --help                   Print help

``Total Items`` counts the items stored, and ``Next Item Id`` is the id the next
insert gives out. They are equal unless a crash stopped an insert between taking
ids and writing the items. ``info`` then reports the gap, and those ids are
never reused.

cleanup-queries
---------------

Unfinished queries saved by ``search`` stay in the index until someone resumes
them. This erases the ones saved more than ``--older-than-hours`` ago, for
example from a scheduled job:

.. code-block:: bash

   # Erase anything saved more than a day ago.
   ecp cleanup-queries my_index.zarr --older-than-hours 24

.. code-block:: text

   Usage: ecp cleanup-queries [OPTIONS] --older-than-hours <OLDER_THAN_HOURS> <INDEX_PATH>

   Arguments:
     <INDEX_PATH>  The index to clean up

   Options:
         --older-than-hours <OLDER_THAN_HOURS>
             Erase queries saved more than this many hours ago
         --with-logging
             Log this run to a JSONL file
         --log-dir <LOG_DIR>
             Directory for the log file, ecp_logs/ if not set
         --log-level <LOG_LEVEL>
             Log verbosity. trace also logs every node visited during search [default: info] [possible values: off, error, warn, info, debug, trace]
     -h, --help
             Print help (see more with '--help')

