# Release Audit

Date: 2026-02-18

## Scope

Comprehensive correctness and safety audit of the full librarian codebase at v1.0.1 (commit 7b4c4e4).

## Round 1 — Key issues identified and fixed

1. **P0: `chunk_preview` UTF-8 panic** — `&chunk_text[..200]` in `src/commands/query.rs` slices at byte offset, panicking on multi-byte characters at the boundary. Fixed: use `char_indices()` for safe boundary.
2. **P1: Regex compiled per-call in parser hot paths** — RST, AsciiDoc, Jupyter parsers compiled `Regex::new().unwrap()` on every function call. Fixed: `LazyLock` statics.
3. **P1: `store_document_links` non-transactional** — DELETE + INSERT loop without transaction boundary; crash mid-op loses all links. Fixed: wrapped in `pool.begin()`/`tx.commit()`.
4. **P1: `total_chunks_searched` misleading** — Field held post-filter result count, not actual chunks searched. Fixed: added `result_count` field, `total_chunks_searched` now holds pre-filter Qdrant result count.
5. **P2: MCP `rag_analytics` days/limit unclamped** — Schema declared max 365 but handler didn't enforce. Fixed: `.clamp(1, 365)` for days, `.clamp(1, 100)` for recent_limit.
6. **P2: `char_pos` tracking inaccurate** — RST and AsciiDoc parsers only counted last push element, not full content. Fixed: compute length before push.
7. **P2: Detection.rs per-page regex compilation** — 7 regex compilations per page analysis. Fixed: `LazyLock` statics.

## Round 2 — Additional issues identified and fixed

1. **P1: `parse_html` CSS URL quote-strip panic** — Single-char quoted value `"` causes `raw[1..0]` panic. Fixed: added `raw.len() >= 2` guard in `src/parse/html.rs`.
2. **P2: `search_offset` overflow in HTML heading lookup** — Repeated heading lookup failures compound offset past text length, causing slice panic. Fixed: clamp to `.min(doc.text.len())`.
3. **P2: SPA/auth/JS-route regex per-page** — 8 SPA patterns, 2 auth patterns, 3 JS route patterns compiled per crawl page. Fixed: `LazyLock` statics (`SPA_ROOT_RES`, `AUTH_FORM_RE`, `AUTH_PASSWORD_RE`, `JS_ROUTE_RES`).
4. **P2: `should_follow_link` date regex per-call** — `Regex::new(r"/\d{4}/\d{2}/\d{2}/")` compiled per URL. Fixed: `LazyLock` static `DATE_URL_RE` in `src/crawl/mod.rs`.
5. **P2: CSS background-image regex per-document** — Compiled in `parse_html` per call. Fixed: `LazyLock` static `CSS_BG_IMAGE_RE` in `src/parse/html.rs`.
6. **P2: `chunk_text.len() as i32` truncation** — Silent truncation in `Chunk::new_media` and `Chunk::new_with_modality`. Fixed: `i32::try_from(...).unwrap_or(i32::MAX)`.
7. **P2: Reindex run completion silently dropped** — `let _ = db.complete_ingestion_run(...)` discards DB errors. Fixed: `if let Err(e) = ... { warn!(...) }`.
8. **P2: `registry_sync` regex per-call + nested unwrap** — `apply_type_to_path` compiled regex each call with fragile fallback. Fixed: `LazyLock` static `PATH_PLACEHOLDER_RE`.

## Accepted as-is (P3 — low risk)

- `ProgressStyle::with_template().unwrap()` — static template, always valid.
- `MODEL_REGISTRY` `.expect()` in LazyLock init — intentional fail-fast on corrupt bundled data.
- Negative image chunk index wrap for >2B images — practically impossible.
- TODO: concurrent multimodal embedding — optimization opportunity, not a bug.

## Changes made

### Round 1

- `src/commands/query.rs`: UTF-8-safe `chunk_preview` via `char_indices()`. Added `result_count` field. Fixed `total_chunks_searched` semantics. Added 2 UTF-8 boundary tests.
- `src/parse/rst.rs`: 2 regex → `LazyLock` statics. Fixed `char_pos` tracking.
- `src/parse/asciidoc.rs`: 4 regex → `LazyLock` statics. Fixed `char_pos` tracking.
- `src/parse/jupyter.rs`: 2 regex → `LazyLock` statics.
- `src/crawl/detection.rs`: 7 regex → `LazyLock` statics.
- `src/meta/mod.rs`: `store_document_links` wrapped in transaction.
- `src/mcp/tools.rs`: Analytics parameters clamped.

### Round 2

- `src/parse/html.rs`: CSS URL quote-strip length guard. `search_offset` clamped. Background-image regex → `LazyLock` static.
- `src/crawl/detection.rs`: 13 additional regex → `LazyLock` statics (`SPA_ROOT_RES`, `AUTH_FORM_RE`, `AUTH_PASSWORD_RE`, `JS_ROUTE_RES`).
- `src/crawl/mod.rs`: Date URL regex → `LazyLock` static.
- `src/meta/mod.rs`: `i32::try_from` for `char_end` in media/modality chunk constructors.
- `src/commands/reindex.rs`: Run completion error now logged with `warn!`.
- `src/xinference/registry_sync.rs`: Path placeholder regex → `LazyLock` static.

## Verification

All quality gates passed after both rounds:

- `cargo fmt` — clean
- `cargo clippy -- -D warnings` — zero warnings
- `cargo test` — 187 unit + 2 integration tests passing

## Round 3 — Correctness and async safety

1. **P1: MCP server blocking IO on Tokio runtime** — `src/mcp/server.rs` used `stdin.lock().lines()` (synchronous `std::io`) inside `async fn run()`, blocking the Tokio runtime thread and preventing background `tokio::spawn` tasks from progressing. Fixed: replaced with `tokio::io::BufReader::new(tokio::io::stdin()).lines()` via `AsyncBufReadExt`, and async stdout writes via `AsyncWriteExt`.
2. **P2: `SearchFilter.path_prefix` silently ignored** — `to_qdrant_filter()` in `src/store/mod.rs` never generated a Qdrant condition for the `path_prefix` field. If set, results would silently include unfiltered documents. Fixed: added `Condition::matches_text("doc_uri", prefix)` clause. Added 2 unit tests.
3. **P2: `config print --all` flag ignored** — `handle_config_action` in `src/main.rs` destructured `ConfigAction::Print { all: _ }` discarding the flag. Both `config print` and `config print --all` produced identical output. Fixed: when `all=false`, loads user config file and renders it; when `all=true`, renders full defaults.
4. **P2: MCP `notifications/cancelled` no-op** — ACCEPTED. Handler only logs at debug level. Low risk since most MCP operations are short-lived.
5. **P2: `ProgressStyle::with_template().unwrap()`** — ACCEPTED (repeat from Round 1). Static template string, always valid.

### Round 3 Changes

- `src/mcp/server.rs`: Replaced `std::io::{BufRead, Write}` with `tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader}`. Async stdin line reading and stdout writing.
- `src/store/mod.rs`: `to_qdrant_filter()` now generates `matches_text("doc_uri", ...)` condition for `path_prefix`. Added `test_path_prefix_generates_condition` and `test_path_prefix_combined_with_source_ids` tests.
- `src/main.rs`: `handle_config_action` now differentiates `all=true` (full defaults) from `all=false` (loaded user config).

## Round 4 — Blocking I/O in async context

Full codebase audit focusing on correctness and safety. Scanned all modules in `src/`.

### Findings and fixes

1. **P2: Blocking `std::process::Command` in async functions** — `extract_media_metadata()`, `extract_keyframes()`, and `extract_audio_track()` in `src/commands/ingest.rs` used synchronous `std::process::Command` to run ffprobe/ffmpeg inside `async fn`. This blocks the Tokio runtime thread, starving concurrent tasks. Fixed: replaced with `tokio::process::Command` and `.output().await`.

2. **P2: Blocking `std::fs::read` in async functions** — `process_file()`, `process_audio_file()`, and `process_video_file()` used `std::fs::read(path)?` to read file contents inside async functions. Fixed: replaced with `tokio::fs::read(path).await?`.

3. **P2: Blocking `std::fs::create_dir_all` in async functions** — `extract_keyframes()` and `process_video_file()` used `std::fs::create_dir_all()` synchronously. Fixed: replaced with `tokio::fs::create_dir_all(...).await?`.

4. **P2: Blocking `std::fs::remove_dir_all` in async functions** — Two cleanup calls in `process_video_file()` and `cmd_ingest_dir()` used synchronous directory removal. Fixed: replaced with `tokio::fs::remove_dir_all(...).await`.

### Accepted as-is

- `src/xinference/deps.rs`: `std::process::Command` in synchronous functions — correct, not in async context.
- `src/config/mod.rs`: `std::process::Command` in `check_ffmpeg_available()` / `check_ffprobe_available()` — synchronous utility functions, correct as-is.
- `src/embedding_backend.rs`: Linear retry without jitter — minor, not causing issues in practice.
- `src/embed/mod.rs`: Sequential modality processing — known TODO, optimization opportunity not a bug.
- `src/rank/mod.rs`: BM25 tokenizer filters tokens < 3 chars — by design for noise reduction.

### Round 4 Changes

- `src/commands/ingest.rs`: 9 blocking I/O calls converted to async equivalents:
  - 3× `std::process::Command` → `tokio::process::Command` with `.output().await`
  - 3× `std::fs::read()` → `tokio::fs::read().await`
  - 1× `std::fs::create_dir_all()` → `tokio::fs::create_dir_all().await`
  - 2× `std::fs::remove_dir_all()` → `tokio::fs::remove_dir_all().await`

## Round 5 — Setup, configuration, and onboarding

Full audit of init wizard, configuration loading, CLI dispatch, and onboarding flows.

### Findings and fixes

1. **P1: `load_config` calls `process::exit(1)` instead of returning `Err`** — `load_config()` in `src/main.rs` used `eprintln!` + `std::process::exit(1)` when the config file was missing, despite having a `Result<Config>` return type. This bypasses the normal error propagation path, prevents callers from handling the error gracefully, and makes the function untestable. Fixed: replaced with `return Err(Error::Config(...))`.

2. **P2: `handle_ingest` silently discards `--extensions` and `--exclude` CLI flags** — `IngestSource::Dir` destructured `extensions: _` and `exclude: _`, silently ignoring user-provided CLI arguments with no feedback. Users would believe their filters were applied when they were not. Fixed: bound the values and added `tracing::warn!` when either is `Some`.

3. **P2: `build_qdrant_compose` uses deprecated `version: "3.9"` key** — Docker Compose V2 ignores the `version` key and emits a deprecation warning. The auto-generated compose file included it unnecessarily. Fixed: removed the `version` line.

4. **P2: `prompt_select` clears entire terminal history on every keypress** — The init wizard's selection prompt used `terminal::Clear(ClearType::All)` + `cursor::MoveTo(0, 0)`, which erases the user's entire terminal scrollback buffer on every keypress. Fixed: save cursor position once before the loop, then `RestorePosition` + `Clear(FromCursorDown)` on each redraw to only clear the prompt area.

### Accepted as-is

- `qdrant_health_url` assumes gRPC port 6334 maps to REST 6333 — standard Qdrant port convention.
- `run_init_wizard` uses blocking terminal I/O in async context — single-threaded init flow, no concurrent tasks.
- `resolve_interactive` rejects piped input — intentional, requires `--non-interactive` flag.
- `compute_irrelevant_paths` conditional logic — correct for all feature flag combinations.
- `build_qdrant_compose` API key env var handling — correct (checks existence properly).

### Round 5 Changes

- `src/main.rs`: `load_config` returns `Err(Error::Config(...))` instead of `process::exit(1)`. `handle_ingest` warns on ignored `--extensions`/`--exclude` flags. `build_qdrant_compose` removes deprecated `version: "3.9"` key.
- `src/commands/init.rs`: `prompt_select` uses `SavePosition`/`RestorePosition` + `Clear(FromCursorDown)` instead of clearing entire terminal.

## Round 6 — Remaining modules sweep

Full audit of all remaining unaudited modules: classify, embed/http_backend, embed/mod, embedding_backend, store/mod, store/payload, rank, rerank, commands/{prune,sources,status,update,xinference}, chunk, crawl/ssrf, error, progress, lib.

### Findings and fixes

1. **P2: Blocking `std::fs::read` in async embedding backend** — `encode_file_base64()` in `src/embed/http_backend.rs` used synchronous `std::fs::read()` inside an `async fn` context (called from 6 async trait methods). This blocks the Tokio runtime thread while reading potentially large media files (images, audio, video). Fixed: replaced with `tokio::fs::read(path).await` and restructured all 6 callers from sync `.map().collect()` iterator chains to async for-loops (iterators cannot contain `.await`).

2. **P2: Silent deserialization failure in Qdrant payload** — `From<Map<String, Value>> for ChunkPayload` in `src/store/payload.rs` used `.unwrap_or_else(|_| ...)` which silently swallowed `serde_json` deserialization errors. If Qdrant returns an unexpected payload shape (e.g., after schema migration), the error would be invisible. Fixed: added `tracing::warn!("Failed to deserialize Qdrant payload, using defaults: {e}")` to log the error before returning the fallback.

### Accepted as-is

- `embed/mod.rs` `fuse_embeddings` zip truncation — caller validates dimensions match before calling; zip semantics are intentional.
- `embedding_backend.rs` linear retry without jitter — minor (local/LAN backend); jitter is a nice-to-have.
- `commands/prune.rs` path existence check — `dry_run` safeguards are already in place.
- `embed/mod.rs` `embed_unified` returning empty vec for zero inputs — correct short-circuit.

### Round 6 Changes

- `src/embed/http_backend.rs`: `encode_file_base64()` converted from `std::fs::read` to `tokio::fs::read().await`. All 6 async callers (`embed_images`, `embed_multimode`, `embed_audio`, `embed_audio_multimode`, `embed_video`, `embed_video_multimode`) restructured from sync iterator chains to async for-loops.
- `src/store/payload.rs`: `ChunkPayload` deserialization fallback now logs a `tracing::warn!` with the error message.

---

## Round 7 — Duplication Audit

**Scope:** Full codebase scan for non-trivial duplicated logic that risks divergence.

### Findings and fixes

1. **DUP-P1-001: Dimension mismatch check duplicated 4× (fixed)** — The pattern `if embedder.dimension() != store.dimension() { return Err(...) }` was copy-pasted in `cmd_ingest_dir`, `cmd_ingest_url`, `cmd_ingest_sitemap` (all in `src/commands/ingest.rs`) and `cmd_reindex` (in `src/commands/reindex.rs`). The ingest copies used a verbose error message (model, family, dimension source), while reindex used a shorter one. **Fix:** Extracted `pub(crate) fn validate_dimensions(embedder, store, embedding) -> Result<()>` in `ingest.rs` with the verbose error format. All 4 call sites now use the shared helper. Reindex now benefits from the richer diagnostics.

2. **DUP-P1-002: Stale doc cleanup + run completion duplicated 3× (fixed)** — Identical blocks of ~35 lines appeared at the end of `cmd_ingest_dir`, `cmd_ingest_url`, and `cmd_ingest_sitemap`: delete stale documents → loop to remove Qdrant points → build errors Option → `complete_ingestion_run` → info log. **Fix:** Extracted `async fn cleanup_and_complete_run(db, store, source_id, run_id, current_uris, stats, label) -> Result<()>` in `ingest.rs`.  All 3 call sites replaced with a single function call. The `label` parameter preserves the per-command log prefix ("Ingestion" vs "Sitemap ingestion").

### Accepted as-is (justified)

- **`embed_*_in_batches` (3 functions in `src/embed/mod.rs`)** — `embed_in_batches`, `embed_images_in_batches`, `embed_multimode_in_batches` share the same loop-chunk-embed structure but operate on different input types (`&[String]` / text vs `&[ExtractedMedia]` / images vs multimode). Extracting a generic would require complex trait bounds for minimal gain. The functions are simple wrappers (~15 lines each) and the type divergence is the point.
- **Xinference bootstrap in `create_embedder_auto` vs `create_reranker_auto`** — Both call `ensure_xinference_ready` then create an HTTP client, but for different resource kinds (embedding vs reranking). The shared pattern is trivial (~8 lines) and extracting it would over-abstract fundamentally different paths.
- **`HttpEmbedder` 6 embed methods** — Each handles a different modality (text, images, audio, video, multimode variants) with different request types and capability checks. The structural similarity is inherent to the per-modality dispatch pattern.
- **`EmbeddingBackendClient` embed methods** — Same justification as HttpEmbedder; each targets a different API endpoint with different payload shapes.

### Round 7 Changes

- `src/commands/ingest.rs`: Added `validate_dimensions()` and `cleanup_and_complete_run()` helper functions. Replaced 3 inline dimension checks and 3 inline cleanup blocks with calls to these helpers. Net reduction: ~100 lines of duplicated logic.
- `src/commands/reindex.rs`: Replaced inline dimension check with `validate_dimensions()` call. Now uses the same rich error message as ingest paths.
