# Whole-Repository Adversarial Audit

Date: 2026-05-05
Issue: librarian-mf5
Scope: Entire repository at `main`, including Rust source, CLI/MCP paths, docs, GitHub workflows, and bd state.

## Method

This pass used the adversarial-audit workflow: inspect the repository contract, look for correctness and operational failure modes, remediate high-confidence issues directly, then verify with commands that exercise the touched surfaces. The pass also checked the existing audit notes in `docs/AUDIT_REPORT.md` and `RELEASE_AUDIT.md` to avoid reopening stale findings that had already been remediated.

External reference used: GitHub's repository custom instructions documentation, which defines `.github/copilot-instructions.md` for repository-wide instructions and `.github/instructions/*.instructions.md` with `applyTo` frontmatter for path-specific instructions.

## Findings

### F1: MCP search re-resolved embedding config despite cached embedder

Status: Fixed
Severity: High
Files: `src/mcp/server.rs`, `src/mcp/tools.rs`

`McpServer::get_or_init_embedder()` cached only the embedder instance. `rag_search` still called `config.resolve_embedding_config().await` inside `handle_search()` for every search, so a cached MCP embedder could still re-enter Xinference registry/model resolution and fail or pay startup cost per query. This violated the repo convention that MCP embeds lazily and reuses the result across tool calls.

Fix: cache embedding config and embedder together in `CachedEmbedder`, and have MCP search reuse both values. Added a regression test where runtime config is deliberately invalid but the cached config is valid, proving the cached path no longer resolves runtime config.

Evidence:

```text
cargo test cached_search_embedding_skips_runtime_config_resolution
test mcp::tools::tests::cached_search_embedding_skips_runtime_config_resolution ... ok
```

### F2: Current clippy gate failed on the repository

Status: Fixed
Severity: Medium
Files: `src/commands/init.rs`, `src/commands/reindex.rs`, `src/parse/markdown.rs`, `src/mcp/tools.rs`

The repository failed current `cargo clippy --workspace --all-targets --all-features -- -D warnings` before remediation. Failures were mechanical but blocking: collapsible nested `if`, redundant `.into_iter()`, a collapsible markdown parser match, and a test module placed before production items.

Fix: applied the clippy-suggested simplifications and moved the MCP test module to EOF.

Evidence:

```text
cargo clippy --workspace --all-targets --all-features -- -D warnings
Finished `dev` profile [unoptimized + debuginfo]
```

### F3: Release workflow had a slim build lane that CI did not check

Status: Fixed
Severity: Medium
Files: `.github/workflows/ci.yml`

`.github/workflows/release.yml` builds a `--no-default-features` artifact for the slim binary, but CI only checked all-features builds. A slim-only compile break could pass CI and fail during release.

Fix: added a CI `cargo check --workspace --no-default-features` lane immediately after clippy.

Evidence:

```text
cargo check --workspace --no-default-features
Finished `dev` profile [unoptimized + debuginfo]
```

### F4: Documentation markdown fences and onboarding path had drifted

Status: Fixed
Severity: Medium
Files: `README.md`, `CONTRIBUTING.md`

`README.md` left Quick Start's shell fence open through the "What's New" section, making the rendered documentation wrong. `CONTRIBUTING.md` had `cd librarian/librarian` after cloning `librarian.git`, and a malformed closing fence labeled as text.

Fix: closed the README fence before the next heading, corrected the clone directory, and repaired the contributing fence.

Evidence:

```text
awk markdown fence parity check over README.md, CONTRIBUTING.md, CONVENTIONS.md,
RELEASE_AUDIT.md, docs/*.md, .github/*.md, and .github/instructions/*.md
no files reported unbalanced fences
```

### F5: Tracked GitHub agent instructions contained machine-specific absolute paths

Status: Fixed
Severity: Low
Files: `.github/copilot-instructions.md`, `.github/instructions/standards-and-conventions.instructions.md`

Tracked instruction files referenced one GitHub Actions workspace path and one local macOS path for `CONVENTIONS.md`. Those paths are not portable repository instructions. GitHub's Copilot documentation defines repository-scoped instruction files under `.github/` with repository-relative context and `applyTo` globs for path-specific files.

Fix: replaced absolute machine paths with repository-relative wording that points to `CONVENTIONS.md`.

Evidence:

```text
rg -n "CONVENTIONS.md" .github
.github/copilot-instructions.md and .github/instructions/standards-and-conventions.instructions.md point to CONVENTIONS.md without machine-specific absolute paths
```

### F6: bd generated runtime state that was not ignored

Status: Fixed
Severity: Low
Files: `.beads/.gitignore`, `.gitignore`

`bd ready --json` migrated bead state and warned that auto-export's `git add` failed because `.beads/export-state.json` was untracked runtime state. `bd doctor` also warned about outdated runtime ignore patterns.

Fix: added bd runtime/credential/daemon artifacts to `.beads/.gitignore` and ignored `.beads-credential-key` at the repository root.

Evidence:

```text
git status --short --ignored .beads .beads-credential-key
!! .beads/export-state.json
```

## Verification Results

The final gate set for this audit passed locally except for expected bd administrative warnings that remain until the session closeout commits and pushes this work.

```text
cargo fmt --check
passed

cargo clippy --workspace --all-targets --all-features -- -D warnings
passed

cargo check --workspace --no-default-features
passed

cargo test
195 unit tests, 6 CLI integration tests, 2 MCP integration tests, 0 doc-tests passed

scripts/ci/check_legacy_backend.sh
No legacy backend references found.

scripts/ci/verify_cli.sh
35 passed, 0 failed, 0 skipped

awk markdown fence parity check
passed

bd doctor
68 passed, 6 warnings, 0 errors before commit/push closeout
```

## Residual Risk

This audit exercises the compile/test/CLI surfaces locally. It does not prove a live Qdrant plus Xinference end-to-end ingestion and MCP query against real external services. Follow-up `librarian-jmo` tracks that bounded integration lane.
