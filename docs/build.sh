#!/usr/bin/env bash
# Builds the combined docs site: Sphinx (Python API) at the root, rustdoc
# (ecp-core/ecp-cli) nested under rust/.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf docs/_build/html
cargo doc --no-deps -p ecp-core -p ecp-cli
uv run sphinx-build -b html docs docs/_build/html

mkdir -p docs/_build/html/rust
cp -R target/doc/. docs/_build/html/rust/

echo "Site built at docs/_build/html/index.html"
