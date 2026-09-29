Zarr Concepts
=============

An index is stored as `Zarr <https://zarr.dev/>`_, and this page explains
what that means, concept by concept, alongside the actual ``zarrs`` Rust
call ``ecp-core`` makes for each. It is useful both to someone who has
never used Zarr and wants to understand what is on disk, and to anyone
modifying ``ecp-core``'s storage code and meeting ``zarrs``'s own API for
the first time. :doc:`format` covers ecpfs's own array layout in detail;
this page is what to read first.

Store
-----

A Zarr store is where every array and group's bytes actually live. It is
pluggable, backed by anything that can read, write, and list keyed byte
ranges. Every on-disk ecpfs index uses one concrete store,
``FilesystemStore``, opened from a path:

.. code-block:: rust

   let store = Arc::new(FilesystemStore::new(&index_path)?);

``zarrs`` also ships a ``MemoryStore``, an in-memory backend with no disk
I/O at all. ``ecp-core``'s own tests use it for fast, disk-free
fixtures, but no production code path does. Do not confuse it with
``EmbeddingsSource::Memory``, an unrelated in-memory source for a
build's embeddings, holding a plain ``Array2<f32>`` with no Zarr store
involved.

Arrays
------

A Zarr array is a chunked, typed, compressed N-dimensional array, stored
as a set of plain files rather than one binary blob. ``ecp-core`` creates
one with ``ArrayBuilder``, given a shape, a chunk shape, a data type, and
a fill value:

.. code-block:: rust

   let mut builder = ArrayBuilder::new(shape, chunk_shape, data_type, fill_value);
   builder.bytes_to_bytes_codecs(compressor());       // compress each chunk with zstd
   let array = builder.build(store.clone(), path)?;   // Array, still only in memory
   array.store_metadata()?;                           // write its zarr.json to the store

This writes the array's shape, dtype, and chunking, not any values.
Values are written in chunks, covered next.

Reading an existing array back is a plain path lookup:

.. code-block:: rust

   let array = Array::open(store.clone(), path)?;

Chunking
--------

An array's chunk shape is fixed when it is created, derived from
``node_chunk_bytes`` or ``rep_chunk_bytes`` divided into whole rows. A
read or write never touches a whole array. It targets an
``ArraySubset``, and only the chunks that subset overlaps are decoded or
written:

.. code-block:: rust

   array.store_array_subset(subset, embeddings)?;                 // write values
   array.retrieve_array_subset::<Array2<f32>>(subset)             // read that subset back
   array.retrieve_array_subset::<Vec<u32>>(&array.subset_all())   // read the whole array

See :doc:`tuning` for why the chunk size choice matters, and
:doc:`format` for what an underfilled chunk costs on disk, nothing, it
compresses away, versus on a read, the whole chunk still decodes.

Compression
-----------

Every array is compressed with zstd at level 3:

.. code-block:: rust

   pub(super) fn compressor() -> Vec<Arc<dyn BytesToBytesCodecTraits>> {
       vec![Arc::new(ZstdCodec::new(3, false))]
   }

Data types
----------

A Zarr array's own type, ``float32``, ``float16``, ``uint8``, or
``int8``, maps directly to ``EmbeddingDtype`` through a small per-variant
match:

.. code-block:: rust

   let data_type = match dtype {
       EmbeddingDtype::F32 => float32(),
       EmbeddingDtype::F16 => float16(),
       EmbeddingDtype::UInt8 => uint8(),
       EmbeddingDtype::Int8 => int8(),
   };

Groups
------

A Zarr *group* is a named collection of arrays and other groups, roughly
a directory. ``ecp-core`` registers one at every container path, the
store's root, ``info``, ``index_root``, each level, and each node, all
through the same small helper:

.. code-block:: rust

   pub(crate) fn build_group(store: &ReadableWritableListableStorage, path: &str) -> Result<()> {
       GroupBuilder::default()
           .build(store.clone(), path)?  // create the group in memory
           .store_metadata()?;           // persist its zarr.json
       Ok(())
   }

This writes a ``zarr.json`` with ``"node_type": "group"`` at ``path``.
Without it, ``path`` is not a group at all, just a directory prefix two
array paths happen to share.

Reading a group back is the same shape as opening an array:

.. code-block:: rust

   let group = Group::open(store.clone(), path)?;

``ecp-core`` never actually calls this. Its own reader opens every array
by its full path directly, never through a group. The read side is
shown here because it is genuinely useful on its own, for inspecting a
store by hand in a test or a script, even though ecp-core's own code
has no need for it.

Groups exist for the store's whitebox goal. A generic Zarr reader such
as zarr-python can then browse and open the hierarchy on its own,
without already knowing every path in advance.

On disk
-------

Every array and group in the Zarr v3 spec has its own ``zarr.json``
metadata file. ``ecp-core``'s own tests assert on this directly
(``ecp-core/tests/format_version.rs``), including a group's, at the
store's root and at every level and node path. An array whose data fits
in a single chunk along every axis stores that chunk in a file named
exactly ``c``. An array chunked along more than one axis adds one path
segment per axis, so a two-dimensional array's first chunk is ``c/0/0``.

Because the store is just Zarr, any Zarr-compliant reader, such as
`zarr-python <https://zarr.readthedocs.io/>`_, can open, browse, and
read it directly, independently of ecpfs's own loader:

.. code-block:: python

   import zarr
   root = zarr.open_group("my_index.zarr", mode="r")   # open the store as a group
   root.tree()                                          # print the whole hierarchy
   root["lvl_1"]["node_0"]["embeddings"][:]              # read values, group by group

:doc:`format` describes every array ecpfs actually writes, by name.
