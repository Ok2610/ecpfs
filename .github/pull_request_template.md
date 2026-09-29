### Summary

One or two sentences: what this PR does and why.

### What changed

What's different because of this PR? Group it by concern, not by file or
by commit, and name the files each concern touches, for example "the
search retry logic (`search/index/query.rs`)".

### Testing

What new tests did you add, and what do they cover?
(Not just "ran the suite," the gate below already confirms that.)

- [ ] The gate passes locally: 
      * `cargo fmt --check`
      * `cargo clippy --workspace --all-targets -- -D warnings`
      * `cargo test --no-default-features`
      * `uv run pytest tests/`
      * `docs/build.sh` (if `docs/` changed).

### Version

Does this PR need a version bump? See `CONTRIBUTING.md`'s Versioning
section.

- Bug fix or small addition: the last number.
- Breaking change or major feature addition: the middle number.

If the size call isn't obvious either way, say so here rather than
deciding it alone.

---

Commits are signed off (`git commit -s`) per `CONTRIBUTING.md`.
