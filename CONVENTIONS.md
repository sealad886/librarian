# Repository Standards & Conventions

## 1. Scope and Purpose

- Captures non-obvious, repo-specific rules that keep ingestion, status, and MCP behaviors consistent and debuggable.

## 2. Core Conventions

### Ingestion runs record operation + interaction mode

**Status:** REQUIRED  
**Scope:** All ingest/update/reindex code paths (CLI, MCP, background jobs)  
**Rule:** Always invoke `cmd_ingest_dir|url|sitemap` with the appropriate `RunOperation` (`Ingest`, `Update`, `Reindex`) and `interactive` flag. Use `interactive = false` for background/update/reindex flows to avoid blocking prompts; use `interactive = true` for CLI-driven ingestion.  
**Rationale (Why this exists):**  

- Ensures `ingestion_runs.operation` accurately reflects the initiating command for status surfaces.  
- Prevents background/MCP/update paths from hanging on interactive prompts.  
- Keeps per-source state consistent for `rag_status`/`rag_sources`.  
**Examples:**  
- Good: `cmd_ingest_dir(&config, &db, &store, path, name, RunOperation::Update, false)` inside update.  
- Bad: Calling ingest helpers without setting `RunOperation` (records as ingest) or leaving `interactive = true` in background tasks (will prompt and hang).  
**Related Files / Modules:**  
- `src/commands/ingest.rs`  
- `src/commands/update.rs`  
- `src/commands/reindex.rs`

### MCP triggers run asynchronously with fresh connections

**Status:** REQUIRED  
**Scope:** MCP tool handlers in `src/mcp/tools.rs`  
**Rule:** Background MCP triggers (`rag_ingest_source`, `rag_update`, `rag_reindex`) must spawn work with new `MetaDb::connect` / `QdrantStore::connect` instances using the cloned `Config`; do not move long-lived handles into `tokio::spawn`.  
**Rationale (Why this exists):**  

- `QdrantStore` isn’t `Clone`; re-connecting avoids lifetime and Send issues.  
- Prevents the MCP server from blocking while operations run.  
**Examples:**  
- Good: `tokio::spawn(run_update_background(config.clone(), ...))` where the background function reconnects to DB/store.  
- Bad: Capturing shared `&MetaDb`/`&QdrantStore` directly in `tokio::spawn` (lifetime/Send errors, potential blocking).  
**Related Files / Modules:**  
- `src/mcp/tools.rs`

### Chunk writes must use canonical document IDs

**Status:** REQUIRED  
**Scope:** All ingestion flows that write chunks (dir/url/sitemap, updates, tests)  
**Rule:** Always use the `Document` returned by `MetaDb::upsert_document` when writing chunks. The returned `Document` carries the canonical `id` for `(source_id, uri)`; never reuse a freshly generated UUID when the doc already exists.  
**Rationale (Why this exists):**  

- Avoids `FOREIGN KEY constraint failed` on `chunks.doc_id → documents.id` when re-ingesting existing docs.  
- Ensures chunk updates/embeddings attach to the persisted document row.  
- Keeps document/chunk stats accurate across ingest/update/reindex.  
**Examples:**  
- Good: `let doc = db.upsert_document(&doc).await?; process_chunks(... &doc, ...)`  
- Bad: `db.upsert_document(&doc).await?; process_chunks(... &original_doc_with_new_uuid, ...)`  
**Related Files / Modules:**  
- `src/meta/mod.rs`  
- `src/commands/ingest.rs`

### Multimodal indexing is capability-gated

**Status:** REQUIRED  
**Scope:** Configuration validation across crawl/embedding/reranker  
**Rule:** Enabling multimodal crawling (`crawl.multimodal.enabled = true`) requires `embedding.supports_multimodal = true`. Audio/video ingestion is not yet supported and must remain disabled. If `reranker.supports_multimodal = true`, then `reranker.enabled` must also be `true`.  
**Rationale (Why this exists):**  

- Prevents configuration from enabling features unsupported by the current models.  
- Fails fast with clear error messages to avoid silent partial ingestion.  
- Keeps behavior deterministic and observability consistent.  
**Examples:**  
- Good: `embedding.supports_multimodal = true` with `crawl.multimodal.enabled = true` and `include_images = true`.  
- Bad: `crawl.multimodal.enabled = true` while `embedding.supports_multimodal = false` (validation error).  
- Bad: `reranker.supports_multimodal = true` with `reranker.enabled = false` (validation error).  
**Related Files / Modules:**  
- `src/config/mod.rs`  
- `src/config/defaults.rs`

### Image chunks carry modality + media metadata

**Status:** REQUIRED  
**Scope:** Chunk creation, cleanup, and reindex flows (ingest, update, reindex)  
**Rule:** Always set `Chunk.modality = "text"` for text chunks and `Chunk.modality = "image"` (with `media_url`/`media_hash`) for image assets. Use modality-aware helpers (`get_chunks_by_modality`, `delete_chunks_by_modality`) so text cleanup does not delete image chunks.  
**Rationale (Why this exists):**  

- Ensures Qdrant points can be deleted/reindexed correctly for both text and image assets.  
- Prevents text-only cleanup (e.g., when a document shrinks) from erasing image embeddings.  
- Keeps reindex behavior deterministic across modalities.  
**Examples:**  
- Good: `Chunk::new_media(..., media_url, media_hash)` and `delete_chunks_by_modality(doc_id, "image")` before re-embedding images.  
- Bad: Storing images as text chunks without `modality` or letting `delete_chunks_from_index` remove image chunks.  
**Related Files / Modules:**  
- `src/meta/mod.rs`  
- `src/commands/ingest.rs`  
- `src/commands/reindex.rs`

## 3. Rationale and Examples

- See examples embedded within each convention above for concrete good/bad patterns that align status reporting and background execution with run tracking.

## 4. Known Exceptions

- None currently documented.

## 5. Change History (Human-Readable)

- 2026-01-19: Added conventions for RunOperation-aware ingestion and asynchronous MCP triggers with fresh connections.
- 2026-01-19: Added convention for modality-aware image chunks and multimodal cleanup.
