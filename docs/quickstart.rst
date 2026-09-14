Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install ecpfs

Building an index
------------------

.. code-block:: python

   from ecpfs import Builder, Metric, EmbeddingDtype

   builder = Builder(
       index_path="my_index.zarr",
       levels=3,
       metric=Metric.L2,
       is_normalized=False,
       # Omit embedding_dtype to match the source's own dtype (the
       # default). Pass EmbeddingDtype.F32/.F16/.UInt8/.Int8 to force one;
       # forcing a narrower dtype than the source logs a warning, since it
       # loses precision and, for the integer dtypes, truncates fractions
       # and clamps out-of-range values. Every read widens back to f32, so
       # a narrow dtype saves disk and read bandwidth, not memory.
       embedding_dtype=None,
   )
   builder.select_representatives(
       embeddings_file="embeddings.zarr",
       target_cluster_items=100,
       strategy="offset",
       fallback_batch_rows=100_000,
   )
   builder.build(
       embeddings_file="embeddings.zarr",
       fallback_batch_rows=100_000,
   )

Searching an index
--------------------

.. code-block:: python

   import numpy as np
   from ecpfs import Index

   with Index("my_index.zarr") as index:
       query = np.random.rand(128).astype(np.float32)
       items, query_id = index.new_search(
           query=query,
           k=10,
           search_exp=4,
           max_increments=-1,
           exclude_vec=[],
       )
       # items: list[(distance, item_id)]

       # Pull further results for the same query without re-searching from the root:
       more_items = index.get_next_k_items(
           query_id=query_id,
           k=10,
           search_exp=4,
           max_increments=-1,
           exclude_vec=[],
       )
   # The `with` block's __exit__ calls close() automatically, see below.

Adding data to an existing index
---------------------------------

``insert`` routes each new point to its nearest leaf and appends it there,
the same descent search already does. Ids are assigned automatically,
starting at the index's current item count, the same convention the
initial build already uses for its own dataset. ``insert`` returns the
assigned ``(start_id, end_id)`` range (``end_id`` excluded) so the caller
can map its own external ids to them.

.. code-block:: python

   import numpy as np
   from ecpfs import Index

   with Index("my_index.zarr") as index:
       # 2 new rows, matching the index's own dimensionality
       new_embeddings = np.random.rand(2, 128).astype(np.float32) 
       start_id, end_id = index.insert(embeddings=new_embeddings)
       # start_id, end_id = 1000, 1002; the two rows got ids 1000 and 1001.

There's no rebalancing. A leaf that keeps growing just keeps growing, so
search quality degrades gradually as an index accumulates far more
inserts than its original build accounted for. A real rebuild is the only
fix for that currently.

``insert`` is not atomic. A crash partway through can leave the index's
item count ahead of what actually landed on disk, permanently skipping
the unwritten ids rather than reusing or colliding with one already
written. All-or-nothing insert semantics are planned for after 1.0; until
then, do not assume an index survives a crash mid-insert without a gap.

Concurrent inserts and searches on one loaded ``Index`` are safe and
fine-grained. Two operations only serialize when they land on the same
leaf, and a search may briefly see pre-insert (stale) data for a leaf an
insert is concurrently touching rather than wait for it. This holds
across Python threads too. ecpfs releases the GIL during search and
insert, so they run with true multithreading rather than just the
appearance of it. It does not extend across separate processes. Two
independent processes (or two separate ``Index(...)`` handles anywhere)
writing to the same index path can race and corrupt data, so the caller
must ensure only one writer touches a given path at a time.

The caller is responsible for new embeddings already matching the
index's ``metric``/``is_normalized``/dtype convention.

The CLI equivalent bulk-loads a whole file in one process, auto-numbering
new ids the same way:

.. code-block:: bash

   ecp add-data my_index.zarr new_embeddings.h5

Persisting and resuming queries
--------------------------------

An ``Index`` keeps every in-flight query (its position in the tree, its
buffered results so far) in memory. ``close()`` persists whatever's still
in flight to disk before releasing the index, and every method raises
``ValueError`` afterward. Using ``Index`` as a context manager (as above)
calls ``close()`` for you, including when the block raises; call it
directly if you're not using ``with``.

A query a caller never finishes draining survives this way across a
process restart, resume it from a completely new ``Index`` pointed at the
same path, using the ``query_id`` returned by the original ``new_search``:

.. code-block:: python

   from ecpfs import Index

   with Index("my_index.zarr") as index:
       items, query_id = index.new_search(
           query=query, k=2, search_exp=1, max_increments=-1, exclude_vec=[]
       )
   # index.close() already ran; query_id's remaining progress is on disk.

   # ... later, possibly a different process ...
   with Index("my_index.zarr") as index:
       more_items = index.get_next_k_items(
           query_id=query_id, k=2, search_exp=1, max_increments=-1, exclude_vec=[]
       )

A query that's already fully explored, with nothing left to hand back, is
erased rather than persisted, so finished queries don't accumulate on disk
by themselves. One that's *not* finished does accumulate, though, unless a
caller resumes it or ``cleanup_persisted_queries_older_than`` clears it out:

.. code-block:: python

   import datetime
   from ecpfs import Index

   with Index("my_index.zarr") as index:
       # Erase anything persisted more than a day ago.
       cutoff = (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp()
       erased = index.cleanup_persisted_queries_older_than(cutoff)

The equivalent from the CLI, for a scheduled cleanup job:

.. code-block:: bash

   ecp cleanup-queries my_index.zarr --older-than-hours 24

Enabling logging
------------------

.. code-block:: python

   from ecpfs import init_logging

   log_path = init_logging(log_dir="ecp_logs", level="debug")
