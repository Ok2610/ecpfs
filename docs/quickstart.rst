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
       # None stores each file's own dtype. A narrower EmbeddingDtype
       # (F16, UInt8, Int8) saves disk but loses precision, and logs a
       # warning. Every read widens back to f32, so it saves disk, not memory.
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
       # items: list[(score, item_id)], lower score is a better match.
       # See the search parameters page for k, search_exp and max_increments.

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
the same descent search already does. Ids are assigned automatically in
row order, starting at the index's ``next_item_id``. A fresh build sets
that to the number of items it stored, so inserted ids continue straight
on from the build's own. ``insert`` returns the
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
inserts than its original build accounted for. A rebuild is the only fix
for that currently.

``insert`` is not atomic. A crash partway through can leave ``next_item_id``
ahead of what landed on disk, skipping the unwritten ids for good rather than
reusing them or colliding with one already written. ``total_items`` still
counts only what was stored, so do not assume an index survives a crash
mid-insert without a gap in its ids.

Concurrent inserts and searches on one loaded ``Index`` are safe and
fine-grained. Two operations only serialize when they land on the same
leaf, and a search may briefly see pre-insert (stale) data for a leaf an
insert is concurrently touching rather than wait for it. This holds
across Python threads too, because ecpfs releases the GIL during search and
insert. It does not extend across separate processes. Two
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

A query a caller never finishes draining survives this way across a process
restart. Resume it from a new ``Index`` pointed at the same path, using the
``query_id`` the original ``new_search`` returned:

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
----------------

.. code-block:: python

   from ecpfs import init_logging

   log_path = init_logging(log_dir="ecp_logs")  # level defaults to "info"

See :doc:`logging` for the level policy, the record format, and where the
file goes.
