# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is
described in `CONTRIBUTING.md`'s Versioning section, and does not follow
strict semantic versioning while the project stays below `0.10.0`.

## [Unreleased]

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
