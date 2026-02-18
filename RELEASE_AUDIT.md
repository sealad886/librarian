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
