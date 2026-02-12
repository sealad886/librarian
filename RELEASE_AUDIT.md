# Release Audit

Date: 2026-02-12

## Scope

Repository-wide duplication and consistency cleanup focused on ingestion, sitemap SSRF validation, source resolution, point ID derivation, and multimodal embedding pathways.

## Key issues identified

1. Duplicate SSRF validation implementations (`sitemap.rs`, `ingest.rs` tests) diverged from canonical `crawl/ssrf.rs` behavior.
2. Repeated source resolution logic across `update`, `reindex`, and `prune` commands.
3. Repeated Qdrant point UUID derivation patterns in ingest/reindex/prune paths.
4. Duplicated image embedding fusion logic between `ingest` and `reindex`.
5. Near-identical media candidate selectors for audio/video.
6. Additional bug discovered: `MetaDb::get_chunk_by_hash` queried `content_hash` column in `chunks` table (should be `chunk_hash`).

## Changes made

- Consolidated sitemap URL validation onto canonical async `validate_url_ssrf`.
- Removed duplicated sync SSRF helper implementations from `src/crawl/sitemap.rs` and `src/commands/ingest.rs` test-only code.
- Converted sitemap parse internals (`parse_urlset`, `parse_sitemap_index`, `parse_plain_text`) to async and updated tests.
- Added `MetaDb::resolve_sources` and switched `cmd_update`, `cmd_reindex`, and `cmd_prune` to it.
- Added canonical point-ID helpers:
  - `Chunk::point_uuid()`
  - `ChunkRecord::point_uuid()`
- Replaced repeated point-ID derivation call sites in ingest/reindex/prune with helper usage.
- Added shared embed helper `embed_images_with_optional_text_fusion` and used it in ingest + reindex.
- Deduplicated audio/video media candidate selection via `select_media_candidates`.
- Fixed metadata lookup bug in `get_chunk_by_hash` SQL to query `chunk_hash`.

## Verification

Quality gates executed after changes:

- `cargo fmt` ✅
- `cargo clippy -- -D warnings` ✅
- `cargo test` ✅
  - Unit tests: 129 passed
  - Integration tests (`tests/mcp_cli.rs`): 2 passed

No test failures observed.
