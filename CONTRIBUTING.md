# Contributing to ecpfs

ecpfs is an eCP approximate nearest neighbor index with a Rust core
(`ecp-core`), Python bindings (`ecpfs`), and a standalone CLI (`ecp`). The
full documentation, including the architecture and internals pages, is at
https://ok2610.github.io/ecpfs/.

## Development setup

```bash
uv sync --group dev
uv run maturin develop
```

This builds the Rust extension and installs it into a local virtual
environment. `uv run pytest tests/` and `cargo test --no-default-features`
both then work against your local checkout.

## The gate

Every pull request runs three CI jobs: Rust, Python, and docs. Run the
same checks locally before opening one:

```bash
cargo fmt --check  # check formatting, no changes made
cargo clippy --workspace --all-targets -- -D warnings  # lint the whole workspace, warnings fail
cargo test --no-default-features  # run the Rust test suite
uv run pytest tests/  # run the Python test suite
docs/build.sh  # build the Sphinx + rustdoc site
```

`cargo test` needs `--no-default-features` because the `extension-module`
feature, needed for the compiled Python extension, refuses to link
against libpython. Python only provides those symbols at dlopen time,
which breaks the test binary itself.

`docs/build.sh` only matters if you touched `docs/` or a doc comment that
feeds the Rust API reference. Both doc builds fail on warnings, so it also
catches a broken cross-reference or a page missing from a toctree.

## Branches, commits, and pull requests

`main` is protected. Every change lands through a pull request, branched
off `main`. A commit message is 2-3 lines summarizing what changed, not a
full explanation, since code comments already carry the non-obvious "why"
at the point it applies. Open the pull request using the template; it
asks for a summary, what changed grouped by concern, what you tested, and
a version bump if one applies.

## Versioning

Every release before `0.10.0` is a staged checkpoint, not governed by a
fixed rule. From `0.10.0` onward, the last number moves for a bug fix or
a small addition (`0.10.x`). The middle number moves for a breaking
change or a major feature addition, something on the scale of `0.9.95`'s
insert support or `0.9.93`'s concurrent access work (`0.y.x`). Judging
"major" is a call, not a formula, so raise it in the pull request if it
is not obvious either way.

See `CHANGELOG.md` for what actually shipped at each version.

## Code style

`cargo fmt` and `clippy -D warnings` enforce formatting and lints; there
is nothing to configure by hand there. Comments follow one
project-specific rule: a doc comment is 1-3 lines covering what the item
does, its parameters, and its return value. A function with several
distinct phases gets one short inline `//` label per phase instead of a
paragraph up top. For example, `incremental_search` in
`ecp-core/src/search/index/query.rs` labels its phases like this:

```rust
// On the first call, queue every root entry
...
// For an internal node, queue its children
...
// After search_exp leaves, stop or double search_exp
...
// Sort once, whichever way the loop ended
```

Each label sits directly above the code it describes, not gathered into
one block at the top of the function.

## Testing

New functionality needs tests backing it, Rust tests in `ecp-core`'s own
`tests/`/`utests` modules for anything in the core, Python tests in
`tests/` for anything only reachable through the bindings. In your pull
request, say what the new tests cover, not just that the suite passes;
the gate already confirms the suite passes.

## Sign-off (DCO)

Every commit needs a `Signed-off-by` trailer, added automatically with:

```bash
git commit -s
```

This certifies the [Developer Certificate of
Origin](https://developercertificate.org/), meaning you wrote the change
yourself, or otherwise have the right to submit it under this project's
license. It does not transfer any rights beyond what that license already
grants. There is no automated check for this yet, so a pull request
missing it will be asked to amend rather than blocked outright.

## Use of AI

Using AI tools to help write a contribution is fine. Translation and
grammar help is a clear case and actively welcome, since it never touches
what the code actually does. For anything that does touch the code, you
are responsible for understanding what the change does before opening
the pull request, and you are the one who answers review questions, not
a tool. Crediting a tool in a commit's sign-off or trailer is optional,
either way is fine.

## License

Contributions are accepted under this project's existing dual license,
MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.

## Code of conduct

Participation in this project is covered by `CODE_OF_CONDUCT.md`.
