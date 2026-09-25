Index format
============

An index is a directory holding a Zarr store. Every part of it is a zarr array,
so any Zarr reader can inspect it.

Settings
--------

``info/`` holds one scalar per setting.

- ``info/levels`` (uint32): node levels below the root. The last one holds the
  leaves.
- ``info/metric`` (string): ``"L2"`` or ``"IP"``.
- ``info/is_normalized`` (bool, optional): whether every embedding is
  unit-length. An index without it is searched as not normalized.
- ``info/total_items`` (uint32): items stored.
- ``info/next_item_id`` (uint32): the id the next insert gives out. Equal to
  ``total_items`` unless a crash stopped an insert partway.

Representatives
---------------

- ``rep_embeddings``, shape ``(num_representatives, dim)``: every
  representative, the items that every other item is clustered around.
- ``rep_item_ids``, shape ``(num_representatives,)``, uint32: each
  representative's item id.
- ``index_root/embeddings``, shape ``(node_size, dim)``: the root node, which
  holds the first ``node_size`` representatives.

Tree nodes
----------

Level ``N`` of the tree lives under ``lvl_N/``, with one group per node.

- ``lvl_N/node_M/embeddings``, shape ``(n, dim)``: the representatives of node
  ``M``'s children, or on the leaf level, the items' own embeddings.
- ``lvl_N/node_M/node_ids``, shape ``(n,)``, uint32: the child node ids on level
  ``N+1``. Internal levels only.
- ``lvl_N/node_M/item_ids``, shape ``(n,)``, uint32: the item ids. Leaf level
  only.
- ``lvl_N/node_M/border``, shape ``(2,)``, float32: created, never populated.

``n`` differs from node to node, since eCP does not enforce cluster sizes. A node
that received no children during the build is never written, so node ids on a
level can have gaps.

Embeddings arrays are stored as ``float32``, ``float16``, ``uint8`` or ``int8``,
chosen at build time with ``EmbeddingDtype``. Every read widens them to
``float32``.

Chunking
--------

Every array carries its own chunk shape, fixed when the array is created. The
representative arrays take ``rep_chunk_bytes`` and the tree nodes take
``node_chunk_bytes``, both given to ``Builder`` at build time and counted in the
stored dtype.

A node holding fewer rows than its chunk still declares the full shape. That
costs nothing on disk, since the unused part compresses away, but reading the
node decodes the whole chunk. A node holding more rows spans several chunks.

An index keeps the chunk shape it was built with, so changing either setting
affects new indexes only.

Saved queries
-------------

``queries/Q/`` holds query ``Q``'s saved state, written by ``close()`` or when
the query is evicted from memory. It has the query vector (``query``), the queue
of nodes still to visit (``tree_pq_score``, ``tree_pq_is_leaf``,
``tree_pq_level``, ``tree_pq_node_id``), the buffered results (``items_score``,
``items_id``) and the save time in Unix seconds (``persisted_at``).
