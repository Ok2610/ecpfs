# ecpfs

A disk-backed implementation of the extended Cluster Pruning (eCP) approximate
nearest neighbor index, with a Rust core, Python bindings, and a standalone
CLI.

## What is eCP?

Extended Cluster Pruning (eCP) is a hierarchical, cluster-based approximate
nearest neighbor index built for collections too large to search, or even
fit, in memory. A set of vectors is randomly selected from the collection as
"leaders," and every other vector is assigned to its nearest leader, forming
clusters. Because the leaders are used as-is, rather than a centroid
computed after the fact, the index tree is known before any vector is
assigned to a cluster, letting construction use that tree to route each
vector to its cluster instead of comparing it against every leader.

For large collections, the leaders themselves are recursively clustered
into an `L`-level tree, so a query descends the tree rather than scanning
every leader directly. Cluster size is a target item count you choose
rather than something eCP derives from a fixed ratio, typically picked to
line up with a disk read's worth of data. At query time, a search expansion
parameter controls how many clusters are read before returning results,
trading I/O for recall.

This maps directly to this project's parameters: `levels` is `L`,
`target_cluster_items` is the target cluster size, and `search_exp` is the
expansion factor.

## What ecpfs does

ecpfs implements the eCP pipeline as a library for real disk-based use: a
Rust core (`ecp-core`), Python bindings (`ecpfs`, via PyO3), and a CLI
(`ecp`). The "fs" in the name reflects that an index is disk-backed and
lazily loaded rather than held entirely in memory: nodes are read as a
search visits them, with an LRU cache capping how many stay resident.
Incremental search (pulling further results for an already-run query
without restarting it) and this LRU-based memory management are
additions on top of the eCP design, not part of the original algorithm.

The on-disk format is Zarr: each tree node is a group of plain arrays,
deliberately "whitebox" so it stays human-inspectable and easy to extend
later without touching the loader. A second, minimal binary backend is
planned for cases that do not need that extensibility.

Index maintenance (inserting into an already-built index without a full
rebuild) is not implemented; an index is built once and then only searched.
This is intended future work, not an abandoned gap.

## Documentation

- `docs/quickstart.rst` — installation and a build/search walkthrough
- `docs/api.rst` — Python API reference
- `docs/cli.rst` — `ecp` CLI reference
- `docs/build.sh` builds the combined site (Python docs plus the Rust API
  reference); `cargo doc --no-deps -p ecp-core -p ecp-cli` builds just the
  Rust side

## Index format

An index is a directory (currently a Zarr store):

```
info/levels             : int, L
info/metric             : int, 0=L2, 1=IP
info/is_normalized      : bool

rep_embeddings          : shape=(num_representatives, dim)
rep_item_ids            : shape=(num_representatives,), uint32

index_root/embeddings   : shape=(node_size, dim), the top-level leaders

lvl_1/node_M/embeddings : shape=(node_size, dim), for each node M at level 1
lvl_1/node_M/node_ids   : shape=(node_size,), uint32, children at level 2
...
lvl_L/node_M/embeddings : shape=(node_size, dim), leaf-level clusters
lvl_L/node_M/item_ids   : shape=(node_size,), uint32, the collection's item ids
```

## Background

ecpfs builds on ideas first explored in eCP-FS, a file-structure-based eCP
implementation:

```bibtex
@inproceedings{khan2025ecpfs,
  author    = {Khan, Omar Shahbaz and Guðmundsson, Gylfi Þór and Jónsson, Björn Þór},
  title     = {The Curious Case of High-Dimensional Indexing as a File Structure: A Case Study of eCP-FS},
  year      = {2025},
  publisher = {Springer-Verlag},
  address   = {Berlin, Heidelberg},
  doi       = {10.1007/978-3-032-06069-3_24},
  booktitle = {Similarity Search and Applications: 18th International Conference, SISAP 2025, Reykjavik, Iceland, October 1-3, 2025, Proceedings},
  pages     = {303-311},
  numpages  = {9},
  location  = {Reykjavik, Iceland}
}
```

## License

Dual-licensed under MIT or Apache-2.0, at your option. See `LICENSE-MIT` and `LICENSE-APACHE`.