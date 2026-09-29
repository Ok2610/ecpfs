Choosing Parameters
===================

Building an index and running it well both involve choices, not just
defaults to accept blindly. This page covers the ones that matter most:
sizing the tree, storage tradeoffs, a correctness trap, and memory
budgeting. For query-time tuning (``k``, ``search_exp``,
``max_increments``), see :doc:`search-parameters` instead.

``levels`` and ``target_cluster_items``
---------------------------------------

Two numbers decide the tree's shape. ``target_cluster_items`` is the
number of items you want a single leaf node to hold, on average. Dividing
the collection's total item count by it, and rounding up, gives the
number of clusters the whole collection needs:

.. code-block:: text

   total_clusters = ceil(total_items / target_cluster_items)

``levels`` is how many levels the tree descends before reaching those
clusters. The builder spreads ``total_clusters`` evenly across every
level, so each node's fan-out, its number of children, comes from
raising ``total_clusters`` to the power of ``1 / levels``, then
rounding up:

.. code-block:: text

   fan_out = ceil(total_clusters ^ (1 / levels))

Nothing checks whether the result is reasonable. Setting
``target_cluster_items`` far too small for the collection, or ``levels``
far too high for the resulting cluster count, produces a fan-out of 1 or
2 without complaint, a tree that is all depth and no breadth. After a
build, ``ecp info`` prints the actual node sizes, worth checking on a
small test build before committing to the same settings on the full
collection.

Chunk sizes
-----------

``node_chunk_bytes`` (default 512 KiB) and ``rep_chunk_bytes`` (default
8 MiB) size the tree nodes' and representative arrays' chunks. These
defaults came from benchmarking build and read speed directly, not from
a guess, and adding headroom past them measured worse, not better. The
Python docstring's own advice still applies. Measure zarr read speed at
a few sizes on your own data before changing either one. See :doc:`zarr`
for what a chunk actually is and what an underfilled one costs.

``embedding_dtype``
-------------------

Every read widens back to ``f32`` regardless of what is stored, so this
setting saves disk, not memory. Leaving it unset keeps each source
file's own dtype.

- ``F32``: keeps full precision. It is the default when a source is
  already ``f32``.
- ``F16``: halves the disk footprint but loses precision. Choose it
  when the source does not need ``f32``'s full range.
- ``UInt8`` / ``Int8``: store integer data exactly, with no loss, such
  as SIFT-style descriptors (``UInt8``) or scalar-quantized vectors
  (``Int8``).

``metric`` and ``is_normalized``
--------------------------------

``Metric.L2`` is Euclidean distance, lower is closer. ``Metric.IP`` is
inner product, higher is closer, and ecpfs stores it negated so lower
still means closer throughout the API.

Set ``is_normalized`` only when every embedding is truly unit-length.
When it is set, L2 search skips computing each vector's own norm, since
a unit-length vector's norm is already known to be 1. Setting it on
data that is not actually unit-length does not raise an error. It
silently computes the wrong distance instead, since the shortcut's math
assumes something that is not true. This is easy to miss, since the
index still returns results, just in the wrong order.

``memory_limit_bytes``
----------------------

The default is 80% of system RAM, floored to a whole GiB. Inside a
Linux container this reads the container's own memory cap instead of
the host's, so it stays sane under cgroup limits.

On the search side, this caps a cache shared between visited tree nodes
and in-flight queries, split 95% to nodes and 5% to queries. Running
past the cap is not an error. An evicted node is simply reread from
disk on its next visit, and an evicted query is persisted so it can
resume later, see :doc:`quickstart`'s section on persisting queries.

On the build side, 80% of the limit is tracked; the rest is left for
memory this crate does not account for. Of that tracked share, up to
three quarters goes to a node cache for the levels being routed
through, whichever is less; what remains sizes each batch of vectors
processed at once. If the limit is too tight for the vectors' own size,
batches collapse toward one vector each, and a warning says so. That is
a sign to raise the limit or lower the collection's dimensionality, not
something to ignore.

``fallback_batch_rows``
-----------------------

This is a fallback, not the primary control on batch size. When the
embeddings source has its own on-disk chunking, an HDF5 dataset created
with chunking, or any Zarr array, the actual batch size comes from
``memory_limit_bytes``, rounded up to a whole multiple of the source's
own chunk size. ``fallback_batch_rows`` only applies when the source has
no native chunking to round against, such as an unchunked HDF5 dataset,
where it stands in for that missing chunk size directly.

Insert versus a rebuild
-----------------------

Naive insert never rebalances. A leaf node that keeps growing past what the
original build accounted for just keeps growing, and search quality
degrades gradually as it drifts from the tree's original shape. No
threshold in the code flags when this has gone too far, so treat it as
something to watch with your own recall measurements over time, not a
rule ecpfs enforces. A full rebuild is the only fix once it matters.
