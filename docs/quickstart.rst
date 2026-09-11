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
       # default). Pass EmbeddingDtype.F16/.F32 to force one; forcing
       # F16 against an f32 source logs a downcast warning.
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
