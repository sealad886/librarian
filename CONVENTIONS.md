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

## 3. Rationale and Examples

- See examples embedded within each convention above for concrete good/bad patterns that align status reporting and background execution with run tracking.

## 4. Known Exceptions

- None currently documented.

## 5. Change History (Human-Readable)

- 2026-01-19: Added conventions for RunOperation-aware ingestion and asynchronous MCP triggers with fresh connections.
