# Contributing

## Code Quality

```bash
lychee -v .
```

### Rust

Install `mold` before running Cargo commands on Linux. Install cargo-machete
with `cargo install cargo-machete --version 0.9.2 --locked`.

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
cargo machete --with-metadata --skip-target-dir
```

## Code Style & Philosophy

### Typing & Pattern Matching

- Prefer explicit types over raw maps and make invalid states unrepresentable
  where practical.
- Prefer typed variants over string literals when the set of values is known.
- Use exhaustive pattern matching so the compiler verifies all cases.

### Self-Documenting Code

- Use descriptive names that read like documentation.
- Add comments for non-obvious decisions, not to restate the code.
- Test deterministic business outcomes rather than mocks or third-party library
  behavior.
