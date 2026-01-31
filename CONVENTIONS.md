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
**Rule:** Enabling multimodal crawling (`crawl.multimodal.enabled = true`) requires `embedding.model` to be a registry-approved multimodal model (see `src/models.rs`) and must not use late-interaction strategies (e.g., ColPali/ColQwen2). Audio/video ingestion requires ffmpeg/ffprobe in PATH (see separate convention). Multimodal reranker capability is inferred from the model registry (no manual `supports_multimodal` flags).  
**Rationale (Why this exists):**  

- Prevents configuration from enabling features unsupported by the current models.  
- Fails fast with clear error messages to avoid silent partial ingestion.  
- Keeps behavior deterministic and observability consistent.  
**Examples:**  
- Good: `embedding.model = "jinaai/jina-clip-v2"` with `crawl.multimodal.enabled = true` and `include_images = true`.  
- Bad: `crawl.multimodal.enabled = true` with a non-multimodal `embedding.model` (validation error).  
- Bad: `crawl.multimodal.enabled = true` with late-interaction embeddings (validation error).  
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

### Embedding backends must implement the probe contract

**Status:** REQUIRED  
**Scope:** Embedding backend implementations and configuration resolution  
**Rule:** Embedding backends must expose `/capabilities`, `/probe`, `/v1/embed/text`, and `/v1/embed/multimode`. `resolve_embedding_config` must call `/probe` and enforce that returned embeddings and dimensions match the configured model; custom backends must advertise the model id in `/capabilities` when the model list is non-empty.  
**Rationale (Why this exists):**  

- Prevents silent dimension mismatches that corrupt Qdrant vectors.  
- Ensures allowlisted vs custom models are validated consistently.  
- Makes multimodal behavior explicit (joint vs dual-encoder) before ingestion.  
**Examples:**  
- Good: Backend returns `text_embeddings` and (when multimodal) `image_embeddings` or `joint_embeddings` with consistent lengths; config validation fails fast when dimensions differ.  
- Bad: Skipping `/probe` or returning embeddings with varying dimensions.  
**Related Files / Modules:**  
- `src/config/mod.rs`  
- `src/embedding_backend.rs`  
- `src/embed/http_backend.rs`  

### Write operations must use validated Qdrant connections

**Status:** REQUIRED  
**Scope:** All code paths that write to Qdrant (ingest, reindex, update, prune)  
**Rule:** Use `QdrantStore::connect_validated()` for any operation that will write points to Qdrant. This validates collection dimensions match both the current embedding config and the stored metadata from previous sessions.  
**Rationale (Why this exists):**  

- Prevents silent data corruption when embedding model/dimension changes between sessions.  
- Catches configuration drift before vectors are written with wrong dimensions.  
- Provides clear remediation guidance when mismatches are detected.  
- Records collection configuration in SQLite for cross-session consistency checks.  
**Examples:**  
- Good: `QdrantStore::connect_validated(&config, &embedding_config, &db).await?` before ingest/reindex.  
- Bad: Using `QdrantStore::connect()` directly for write operations (bypasses dimension validation).  
**Related Files / Modules:**  
- `src/store/mod.rs`  
- `src/main.rs`  
- `src/mcp/tools.rs`

### Audio/video ingestion requires ffmpeg/ffprobe dependency

**Status:** REQUIRED  
**Scope:** Configuration validation and init wizard when audio/video multimodal is enabled  
**Rule:** When `crawl.multimodal.include_audio = true` or `crawl.multimodal.include_video = true`, the system must verify `ffmpeg` and `ffprobe` are available in `$PATH` before proceeding. Config validation must fail fast with a clear error message if dependencies are missing. The init wizard must prompt for multimedia settings only when multimodal is enabled and check for ffmpeg/ffprobe availability.  
**Rationale (Why this exists):**  

- Audio metadata extraction and transcription require ffprobe for format detection.  
- Video keyframe extraction requires ffmpeg with filter support.  
- Failing fast prevents silent ingestion failures or corrupt metadata.  
- Dependency checks during init ensure the user knows what's required before starting ingestion.  
**Examples:**  
- Good: `which ffmpeg && which ffprobe` check during config validation; init wizard warns if missing.  
- Bad: Enabling `include_audio = true` without checking ffmpeg availability (will fail at runtime).  
**Related Files / Modules:**  
- `src/config/mod.rs`  
- `src/commands/init.rs`  
- `src/commands/ingest.rs`

### Audio/video chunks use derived modalities

**Status:** REQUIRED  
**Scope:** Audio and video ingestion flows (extraction, chunking, embedding)  
**Rule:** Audio files produce text chunks via transcription with `modality = "audio"` containing the transcript text. Video files produce image chunks via keyframe extraction with `modality = "video"` plus optional text chunks from audio track transcription. All multimedia chunks must carry `media_url` pointing to the source asset and `media_hash` for deduplication. Text embeddings are used for transcript chunks; image embeddings for keyframe chunks.  
**Rationale (Why this exists):**  

- Derived modalities (transcript → text embedding, keyframe → image embedding) work with existing embedding backends.  
- Explicit modality tagging enables modality-aware query routing and cleanup.  
- Media URL/hash tracking supports deduplication and asset provenance.  
**Examples:**  
- Good: Audio file → ffprobe metadata → Whisper transcription → text chunks with `modality = "audio"`, `media_url = "file:///path/to/audio.mp3"`.  
- Good: Video file → ffmpeg keyframes → image chunks with `modality = "video"`, plus transcript chunks from audio track.  
- Bad: Storing video keyframes as `modality = "image"` (loses video provenance).  
**Related Files / Modules:**  
- `src/commands/ingest.rs`  
- `src/meta/mod.rs`  
- `src/store/payload.rs`

### Qdrant storage uses named vectors for multimodal

**Status:** REQUIRED  
**Scope:** Qdrant collection creation and vector upsert when multiple modalities are enabled  
**Rule:** When the configuration supports multiple modalities (text + image, or text + image + audio/video), the Qdrant collection must use named vectors (e.g., `"text"`, `"image"`) rather than a single unnamed vector. Each named vector has its own dimension and distance metric. Queries must specify the target vector space via the `using` parameter. The store must validate dimension consistency per vector name before upserting.  
**Rationale (Why this exists):**  

- Different modalities may have different embedding dimensions (e.g., text 768, image 512).  
- Named vectors allow querying specific modalities or combining results.  
- Dimension validation per vector name prevents silent data corruption.  
**Examples:**  
- Good: Collection with `vectors: { "text": { size: 768, distance: "Cosine" }, "image": { size: 512, distance: "Cosine" } }`.  
- Good: Query with `using: "text"` for text search, `using: "image"` for image search.  
- Bad: Single unnamed vector with mixed dimensions (corrupts index).  
**Related Files / Modules:**  
- `src/store/mod.rs`  
- `src/commands/query.rs`  
- `src/config/mod.rs`

### Xinference allowlists come from snapshots

**Status:** REQUIRED  
**Scope:** Model allowlisting and capability checks for Xinference-backed embeddings/rerankers  
**Rule:** Load allowlisted models and capabilities from the versioned snapshots in `resources/xinference/` via `registry_snapshot`; do not query Xinference registry endpoints at runtime. Refresh snapshots only via `cargo xtask xinference-sync` during version bumps.  
**Rationale (Why this exists):**  

- Ensures deterministic allowlists and reproducible config validation.  
- Avoids runtime network calls or registry drift in production.  
- Keeps CI responsible for detecting registry changes on releases.  
**Examples:**  
- Good: `registry_snapshot(RegistryType::Embedding)` inside `src/models.rs` for allowlists.  
- Bad: Calling `/v1/model_registrations` at startup to validate `embedding.model`.  
**Related Files / Modules:**  
- `src/models.rs`  
- `src/xinference/registry.rs`  
- `src/xinference/registry_sync.rs`  
- `src/bin/xtask.rs`  
- `resources/xinference/*.json`

## 3. Rationale and Examples

- See examples embedded within each convention above for concrete good/bad patterns that align status reporting and background execution with run tracking.

## 4. Known Exceptions

- None currently documented.

## 5. Change History (Human-Readable)

- 2026-01-19: Added conventions for RunOperation-aware ingestion and asynchronous MCP triggers with fresh connections.
- 2026-01-19: Added convention for modality-aware image chunks and multimodal cleanup.
- 2026-01-19: Updated multimodal gating to use the model registry and reject late-interaction ingestion.
- 2026-01-20: Added embedding backend probe contract convention.
- 2026-01-30: Added convention for validated Qdrant connections on write operations.
- 2026-01-30: Added conventions for audio/video ffmpeg dependency, derived modalities, and named vectors storage.
- 2026-02-02: Added Xinference registry snapshot rule for deterministic allowlists and version-bump sync.
