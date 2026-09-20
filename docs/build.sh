#!/usr/bin/env bash
# Builds the combined docs site: Sphinx (Python API) at the root, rustdoc
# (ecp-core/ecp-cli) nested under rust/. Both fail on warnings, so a broken
# link or a page missing from the table of contents stops the build.
#
# cli.rst holds `ecp --help` output. Refresh it with docs/gen_cli.py after
# changing any CLI help text; it is a separate step since it rewrites the page.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf docs/_build/html
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p ecp-core -p ecp-cli
uv run sphinx-build -W -b html docs docs/_build/html

mkdir -p docs/_build/html/rust
cp -R target/doc/. docs/_build/html/rust/

echo "Site built at docs/_build/html/index.html"
