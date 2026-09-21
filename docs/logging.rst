Logging
=======

Optional, file-based logging shared by the Python bindings and the CLI. One
log file is created per process (one Python interpreter, one ``ecp`` CLI
invocation, or your own Rust program embedding ``ecp-core``). The file
format is JSONL, which stores a log line as one JSON object per line,
making it easy to read with ``jq`` or load with any JSON-lines reader.

Turning it on
-------------

From Python:

.. code-block:: python

   from ecpfs import init_logging

   log_path = init_logging(log_dir="ecp_logs", level="info")

From the CLI, on any subcommand:

.. code-block:: bash

   ecp search my_index.zarr query.h5 --with-logging --log-dir ecp_logs --log-level info

Both default to ``ecp_logs/`` in the current directory and the ``info``
level. ``init_logging`` returns the file's path; the CLI prints it to
stderr.

Levels
------

Six levels, each including everything the ones before it show: ``off``,
``error``, ``warn``, ``info`` (default), ``debug``, ``trace``.

``off``
   Still creates the log file (so ``init_logging``/``--with-logging``
   returns/prints a path), but logs nothing to it.

``error``
   The operation failed. Only logged when a failure is not returned as a
   Result or exception for the caller to assess.

``warn``
   The operation finished, but the result may be worth a second look, such
   as a narrower ``embedding_dtype`` than the source data or a query id
   that wasn't found.

``info``
   Logs one line per user-level operation, such as loading an index or
   running a search.

``debug``
   Logs internal function details, such as batch information during a
   build, or every search call's parameters.

``trace``
   Logs per-node details during a search, such as a cache hit versus a
   disk read, or how many items a leaf contributed. The most detailed
   level, and the most expensive to leave on for a large search.

Record format
-------------

Every line is one JSON object with four fields, for example:

.. code-block:: json

   {"timestamp":"2026-09-21T12:28:09.815337Z","level":"INFO","target":"ecp_core::search::index::query","message":"search: query_id=0 k=3 leaves_scanned=2 items_returned=3"}

``timestamp``
   RFC 3339, UTC, with sub-second precision.

``level``
   Upper case (``INFO``, ``DEBUG``, ...), regardless of how you spelled it
   in ``init_logging``/``--log-level`` (lower case there, e.g. ``"info"``).

``target``
   The Rust module that logged the line, e.g. ``ecp_core::build::builder``
   or ``ecp_core::search::index::query``. Useful for filtering to one area
   of the code.

``message``
   Plain text, not itself further structured. Parameter values are
   interpolated directly into it (``query_id=0 k=3 ...``), not broken out
   into their own JSON fields.


One file per process
--------------------

One file per process, shared by every thread in it; concurrent writes never
interleave or corrupt a line. A new process gets a new file, even at the
same ``log_dir``, and only the first ``init_logging``/``--with-logging``
call in a process does anything, later calls return the same path. Files
are named ``{start_time}-{6_hex_chars}.jsonl``, for example
``20260921T122809Z-05f388.jsonl``, so sorting by filename sorts them
chronologically, and nothing rotates a file once it's created.

Reading a log
-------------

Every line is independently valid JSON, so ordinary line-oriented tools work
directly on the file, with no JSONL-aware library needed: ``tail``/``head``
for a window of it, ``wc -l`` for how many records it holds, ``jq`` for
filtering and reshaping.

The last 20 records:

.. code-block:: bash

   tail -n 20 ecp_logs/*.jsonl | jq .

Only the errors, across every file in the directory:

.. code-block:: bash

   jq -r 'select(.level=="ERROR") | .message' ecp_logs/*.jsonl

The same two things from Python, one ``json.loads`` per line and no extra
dependency:

.. code-block:: python

   import json
   from pathlib import Path

   records = [
       json.loads(line)
       for path in sorted(Path("ecp_logs").glob("*.jsonl"))
       for line in path.read_text().splitlines()
   ]

   last_20 = records[-20:]
   errors = [r for r in records if r["level"] == "ERROR"]
