# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is
described in `CONTRIBUTING.md`'s Versioning section, and does not follow
strict semantic versioning while the project stays below `0.10.0`.

## [Unreleased]

## [0.9.99] - 2026-09-29

### Added

- `CONTRIBUTING.md`, covering development setup, the gate, branch and
  pull request conventions, versioning, code style, testing, DCO
  sign-off, and AI use.
- `CODE_OF_CONDUCT.md`, the Contributor Covenant.
- GitHub issue forms for bug reports and feature requests, and a pull
  request template.
- `docs/zarr.rst`, explaining Zarr's own concepts alongside the `zarrs`
  Rust calls `ecp-core` makes for each.
- `docs/tuning.rst`, covering build-time and runtime parameter choices:
  `levels` and `target_cluster_items`, chunk sizes, `embedding_dtype`,
  `metric` and `is_normalized`, `memory_limit_bytes`, and when to
  rebuild instead of continuing to insert.

### Changed

- Every docs page title is now Title Case.
- Bare "leaf"/"leaves" is now "leaf node"/"leaf nodes" throughout the
  docs and the `ecp search` help text.
- **Breaking:** the `leaves_scanned` JSONL log field is now
  `leaf_nodes_scanned`.

### Fixed

- `zarr-python` could not browse or open anything below an index's
  root, since `ecp-core` registered arrays but never groups. A group
  is now registered at every container path (root, `info`,
  `index_root`, each level, each node).

## Before this changelog

Versions before `0.9.99` (`v0.9.5` through `v0.9.98`) predate this file.
See the repository's tags for that history.
