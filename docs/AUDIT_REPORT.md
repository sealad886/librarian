# Librarian Codebase Audit Report

**Date:** 2026-02-03  
**Scope:** Full codebase audit — research only, no code changes  
**Version:** 1.0.1  
**Auditor:** GitHub Copilot (Claude Opus 4.6)

---

## 1. Executive Summary

Librarian is a well-engineered Rust CLI tool and MCP server for local RAG (Retrieval Augmented Generation) over documentation. It indexes local directories, web pages, and sitemaps into Qdrant vector storage with SQLite metadata tracking, and exposes search via CLI and MCP-over-stdio. The codebase is ~15,000 lines of Rust across 40+ source files.

**Strengths:**

- Clean module separation with clear data flow boundaries
- Comprehensive multimodal pipeline (text, images, audio, video)
- Robust dimension validation preventing silent Qdrant corruption
- Production-quality SSRF protection and robots.txt compliance
- Smart content hashing for incremental updates
- Canonical document ID convention preventing FK violations
- Deterministic model registry via versioned snapshots
- Thorough configuration validation with fail-fast error messages

**Weaknesses:**

- Code block integrity during chunking is declared but not enforced
- BM25 hybrid scoring uses IDF=1.0 (effectively only term frequency)
- MCP creates a fresh embedder per search call — significant latency
- No document-to-document relationship tracking or cross-references
- No query routing, intent detection, or comprehension tools
- Search filter construction limited to single-value conditions
- HTML heading position detection uses unreliable `text.find()` heuristic
- Test coverage focused on unit tests; integration testing is minimal

**Overall Assessment:** The system is a solid v1.0 RAG indexer with good engineering fundamentals. The primary gaps are in *retrieval intelligence* — the system can find text chunks but cannot reason about document relationships, navigate hierarchies, or answer questions that span multiple documents.

---

## 2. Module Map

| Module | Files | Lines | Health | Notes |
|---|---|---|---|---|
| **CLI / Entry** | main.rs | ~1,463 | ✅ Good | Complete clap-derive CLI, clean dispatch |
| **Config** | config/ (2 files) | ~2,635 | ✅ Good | Extensive validation, multimodal gating |
| **Ingestion** | commands/ingest.rs | ~2,805 | ✅ Good | Full multimodal pipeline, well-structured |
| **Parsing** | parse/ (4 files) | ~1,163 | ⚠️ Fair | HTML heading position is approximate |
| **Chunking** | chunk/ (2 files) | ~498 | ⚠️ Fair | Code block preservation declared but unused |
| **Embedding** | embed/ (2 files), embedding_backend.rs | ~1,125 | ✅ Good | Clean trait design, dimension validation |
| **Store** | store/ (2 files) | ~1,127 | ✅ Good | Validated connections, orphan detection |
| **Ranking** | rank/ (1 file) | ~273 | ⚠️ Fair | BM25 is effectively non-functional |
| **Reranking** | rerank/ (2 files) | ~200+ | ✅ Good | Clean trait, disabled by default |
| **Metadata** | meta/ (2 files) | ~1,622 | ✅ Good | WAL mode, migrations, canonical ID pattern |
| **MCP** | mcp/ (4 files) | ~850+ | ⚠️ Fair | Fresh embedder per call, no comprehension tools |
| **Crawl** | crawl/ (7 files) | ~1,800+ | ✅ Good | SSRF, rate limiting, SPA detection |
| **Xinference** | xinference/ (5 files) | ~1,500+ | ✅ Good | Zero-config, snapshot-based allowlists |
| **Models** | models.rs | ~411 | ✅ Good | OnceLock registry, multimodal strategy types |
| **Commands** | commands/ (10 files) | ~4,500+ | ✅ Good | Init wizard, prune, update, status |
| **Error** | error.rs | ~110 | ✅ Good | Actionable error messages with remediation |
| **Tests** | tests/ | ~100 | ⚠️ Fair | 1 integration test (MCP), good unit tests |

---

## 3. Detailed Findings

### 3.1 Architecture & Data Flow

The system follows a clean pipeline architecture:

```text
 CLI/MCP → Source Resolution → Crawl/Read → Parse → Chunk → Embed → Qdrant + SQLite
                                                                          ↑
 Query → Embed Query → Search Qdrant → Rank (vector ± BM25) → Rerank → Results
```

**Finding A-1: No feedback loop from queries to ingestion.** There is no mechanism to identify poorly-performing sources, frequently-missed queries, or "dead" chunks that never match any query. This limits operational observability.

**Finding A-2: No streaming ingestion.** Each ingestion command processes all files/pages in batch. For very large corpora (10k+ documents), this means long-running, all-or-nothing operations. The update command helps but only re-runs the full source.

**Finding A-3: Named vectors convention documented but not implemented.** [CONVENTIONS.md](../CONVENTIONS.md) requires named vectors for multimodal Qdrant storage, but [src/store/mod.rs](../src/store/mod.rs#L91) uses a single unnamed vector via `VectorParamsBuilder::new(dimension as u64, Distance::Cosine)`. This means text and image embeddings share the same vector space with the same dimension requirement.

### 3.2 Configuration

**Reviewed:** [src/config/mod.rs](../src/config/mod.rs), [src/config/defaults.rs](../src/config/defaults.rs)

**Finding C-1: Comprehensive validation.** Config validation covers chunk bounds, score ranges, model allowlisting, multimodal gating, and ffmpeg dependency checks. This is well above typical for a v1.0 project.

**Finding C-2: Dimension resolution priority chain is clear.** The resolution order (config.embedding.dimension → custom backend → Xinference model spec → probe) is explicitly documented and implemented in `resolve_embedding_config()`.

**Finding C-3: No config file versioning.** The TOML config has no version field. If the schema changes in a future release, there's no migration path for existing config files. The init wizard handles this partially (merge option), but automated upgrades would need schema versioning.

**Finding C-4: ffmpeg dependency is checked at validation time.** When `include_audio` or `include_video` is true, the config validator shells out to check `which ffmpeg` and `which ffprobe`. This is good fail-fast behavior but blocks config loading if ffmpeg is temporarily unavailable.

### 3.3 Ingestion

**Reviewed:** [src/commands/ingest.rs](../src/commands/ingest.rs) (2,805 lines)

**Finding I-1: Well-structured multimodal pipeline.** The ingestion pipeline handles text files, web pages, audio files (via Whisper transcription), and video files (via keyframe extraction). Image candidates are scored by relevance heuristic and deduplicated by perceptual hash.

**Finding I-2: Image relevance scoring is heuristic-based.** The `score_image_candidate()` function uses alt text overlap with headings, URL keyword matching, and other signals. This is reasonable for v1.0 but may miss contextually important images or include decorative ones.

**Finding I-3: Audio/video web crawl infrastructure exists but is unused.** `select_audio_candidates()` and `select_video_candidates()` are implemented and marked `#[allow(dead_code)]`. They're infrastructure for future web crawl audio/video support. Only local file ingestion processes audio/video currently.

**Finding I-4: Source overlap detection is thorough.** Both directory path overlap and URL path overlap detection exist with clear user-facing messages. Interactive mode allows resolution of conflicts.

**Finding I-5: Stale document cleanup has a potential race condition.** `delete_stale_documents()` deletes docs not in the current URI list, then separately deletes their Qdrant points. If the process crashes between these operations, orphan points remain. The prune command handles this, but it's worth noting.

### 3.4 Parsing

**Reviewed:** [src/parse/mod.rs](../src/parse/mod.rs), [src/parse/html.rs](../src/parse/html.rs), [src/parse/markdown.rs](../src/parse/markdown.rs), [src/parse/text.rs](../src/parse/text.rs)

**Finding P-1: HTML heading position detection is unreliable.** [html.rs](../src/parse/html.rs) approximates heading positions using `text.find(&heading_text)`, which returns the position of the *first occurrence* of that text in the document. Documents with repeated heading text (common in API docs with "Parameters", "Returns", etc.) will have incorrect heading-to-position mappings.

**Finding P-2: Markdown images are not extracted as media.** [markdown.rs](../src/parse/markdown.rs) extracts links via `Tag::Link` events but does not handle `Tag::Image` events. Markdown images (`![alt](url)`) are not added to `ParsedDocument.media`. This means multimodal image ingestion doesn't work for markdown files.

**Finding P-3: No frontmatter parsing.** Neither HTML nor Markdown parsers extract YAML/TOML frontmatter metadata. Many documentation sites use frontmatter for title, description, tags, and categories. This metadata is lost during parsing.

**Finding P-4: No RST, AsciiDoc, or Jupyter Notebook support.** Content type detection only handles HTML, Markdown, and plain text. RST (common in Python docs), AsciiDoc (common in Java/enterprise docs), and Jupyter notebooks are treated as plain text.

### 3.5 Chunking

**Reviewed:** [src/chunk/mod.rs](../src/chunk/mod.rs), [src/chunk/boundaries.rs](../src/chunk/boundaries.rs)

**Finding CH-1: Code block integrity is declared but not enforced.** [boundaries.rs](../src/chunk/boundaries.rs) contains `find_code_blocks()` and `is_in_code_block()` functions, but `find_break_points()` in [mod.rs](../src/chunk/mod.rs#L81) never calls them. Break points can be placed inside code blocks, potentially splitting code snippets across chunks.

**Finding CH-2: Heading context propagation is well-designed.** Each chunk receives the heading hierarchy at its starting position via `headings_at_position()`. This means chunks carry the section context they belong to, which is valuable for search relevance.

**Finding CH-3: Overlap implementation is sound.** Chunk overlap uses the configured `overlap_chars` to include trailing context from the previous chunk in the next one. The overlap start position search correctly walks backwards to find a paragraph, sentence, or word boundary.

**Finding CH-4: Content hashing enables efficient incremental updates.** Blake3 hashing of (doc_hash + chunk_text) creates deterministic chunk IDs. Combined with the `existing_hashes` check in `process_chunks()`, unchanged chunks are never re-embedded.

### 3.6 Embedding & Search

**Reviewed:** [src/embed/mod.rs](../src/embed/mod.rs), [src/embed/http_backend.rs](../src/embed/http_backend.rs), [src/embedding_backend.rs](../src/embedding_backend.rs)

**Finding E-1: Clean trait-based embedding architecture.** The `Embedder` trait with methods for text, images, audio, video, and unified embedding is well-designed. Both Xinference and HTTP backends implement it consistently.

**Finding E-2: Dimension validation at multiple layers.** Dimensions are checked at: config load time, embedder creation, before every batch write, and at Qdrant collection validation. This defense-in-depth approach prevents silent corruption.

**Finding E-3: Image embedding fusion is available but limited.** `embed_images_with_optional_text_fusion()` supports joint-input (text+image together) and dual-encoder (separate embeddings averaged) strategies. However, the fusion is simple averaging — more sophisticated attention-based fusion isn't available.

**Finding E-4: Sequential modality processing in `embed_unified`.** The code has a TODO noting that `embed_unified` processes modalities sequentially rather than using `tokio::try_join!`. For documents with both text and image content, this is suboptimal.

### 3.7 Query Pipeline

**Reviewed:** [src/commands/query.rs](../src/commands/query.rs), [src/rank/mod.rs](../src/rank/mod.rs), [src/rerank/mod.rs](../src/rerank/mod.rs)

**Finding Q-1: BM25 hybrid scoring uses IDF=1.0.** [rank/mod.rs](../src/rank/mod.rs) implements BM25 with `idf = 1.0`, making it effectively just a term frequency counter. Without corpus-level inverse document frequency statistics, the BM25 component cannot distinguish common terms from rare, discriminative ones. The comment in code acknowledges this: "Simplified IDF - for single-doc scoring, IDF is less meaningful."

**Finding Q-2: Over-fetch strategy is reasonable.** The query fetches `k * 2` results from Qdrant and then applies ranking/reranking to select the top `k`. This provides enough headroom for post-retrieval ranking to improve results.

**Finding Q-3: No query classification or routing.** Every query follows the same path: embed → search → rank. There's no detection of query intent (e.g., "what is X" vs "how to X" vs "compare X and Y"), no multi-hop retrieval, and no query expansion.

**Finding Q-4: Reranking partitions by text length.** Text chunks shorter than 50 characters are excluded from reranking and placed at the end of results. This is sensible since cross-encoder rerankers need sufficient text context.

**Finding Q-5: Search filter construction is limited.** [store/mod.rs](../src/store/mod.rs#L741) supports filtering by single `source_id` and single `source_type` only. Multi-value OR filters (search in sources A or B) are not supported even though the `SearchFilter` struct accepts `Vec<String>`.

### 3.8 MCP Server

**Reviewed:** [src/mcp/server.rs](../src/mcp/server.rs), [src/mcp/tools.rs](../src/mcp/tools.rs)

**Finding M-1: Fresh embedder created per search call.** [tools.rs](../src/mcp/tools.rs) calls `create_embedder_auto()` (which may start Xinference, launch models, and probe dimensions) for every `rag_search` invocation. This adds significant latency to each query. The embedder should be cached or shared.

**Finding M-2: No comprehension tools.** The MCP exposes only CRUD/search operations: `rag_search`, `rag_sources`, `rag_status`, `rag_ingest_source`, `rag_update`, `rag_reindex`. There are no tools for document summarization, cross-document comparison, concept extraction, or hierarchy navigation.

**Finding M-3: Background operations are fire-and-forget.** `rag_ingest_source`, `rag_update`, and `rag_reindex` spawn background tasks and immediately return success. There's no way via MCP to check the progress or status of a running background operation.

**Finding M-4: MCP protocol is well-implemented.** JSON-RPC 2.0 over stdio with proper protocol version negotiation (2024-11-05), tool listing, and error responses. The implementation is clean and correct.

### 3.9 Metadata & State

**Reviewed:** [src/meta/mod.rs](../src/meta/mod.rs), [src/meta/schema.rs](../src/meta/schema.rs)

**Finding DB-1: Schema is well-designed for the current scope.** Tables for sources, documents, chunks, ingestion_runs, and collection_config cover the operational needs. Indexes are appropriate (source_id, content_hash, chunk_hash, qdrant_point_id).

**Finding DB-2: No document relationship tables.** There are no tables for document-to-document links, cross-references, shared concepts, or section hierarchies. The schema supports flat retrieval but cannot answer "what documents reference this one?" or "what's the parent section of this chunk?"

**Finding DB-3: Migration system is minimal but functional.** Schema versioning uses a `schema_version` table with manual migration functions (`migrate_v0_to_v1`). This works for now but will need a more robust migration framework as the schema evolves.

**Finding DB-4: Canonical document ID pattern is well-tested.** There's an explicit regression test (`test_reingest_document_uses_canonical_id_for_chunks`) verifying that `upsert_document()` returns the original document ID on re-ingestion. This prevents FK violations when chunks reference document IDs.

**Finding DB-5: WAL mode and connection pooling are properly configured.** SQLite runs with WAL journal mode and `NORMAL` synchronous mode for good concurrency and performance. Pool size is capped at 5 connections.

### 3.10 Error Handling

**Reviewed:** [src/error.rs](../src/error.rs)

**Finding ER-1: Error variants are specific and actionable.** `DimensionMismatch` includes the stored/expected dimensions plus a remediation string. `ConfigCollectionConflict` includes both old and new model names. This is excellent UX for CLI tools.

**Finding ER-2: Error propagation is consistent.** The codebase uses `?` operator throughout with the custom `Result<T>` alias. There are no `unwrap()` calls in production code paths (only in tests and string formatting).

**Finding ER-3: Missing error variants for Qdrant timeouts/retries.** The `EmbeddingBackendClient` has retry logic, but Qdrant operations have no retries. Transient network errors during `upsert_points` or `search` will immediately fail the operation.

### 3.11 Tests

**Reviewed:** [tests/mcp_cli.rs](../tests/mcp_cli.rs), inline `#[cfg(test)]` modules

**Finding T-1: Good unit test coverage for core types.** MetaDb has comprehensive tests for source CRUD, document upsert with canonical ID preservation, chunk operations, migration idempotency, and collection config. Store module tests dimension mismatch rejection and filter construction.

**Finding T-2: Only 1 integration test.** [tests/mcp_cli.rs](../tests/mcp_cli.rs) tests MCP server initialization and `--help` output. There are no integration tests for the full ingestion pipeline, query pipeline, or update/prune workflows.

**Finding T-3: No tests for parsing edge cases.** HTML parsing of malformed documents, markdown with complex nesting, and content type detection are not tested directly. The parse module has no `#[cfg(test)]` block.

**Finding T-4: Ingestion tests cover multimodal selection well.** Image candidate selection, CSS background toggling, late-interaction rejection, SVG filtering, and perceptual deduplication all have unit tests.

### 3.12 Documentation

**Finding D-1: README.md is comprehensive.** Installation, architecture diagram, quick start, MCP configuration, and all CLI commands are documented with examples.

**Finding D-2: CONTRIBUTING.md covers development setup.** Prerequisites, project structure, code style, and testing instructions are present.

**Finding D-3: CONVENTIONS.md is thorough and well-maintained.** Nine conventions with status, scope, rule, rationale, and examples. Change history tracks evolution. This is unusually good for a project of this size.

**Finding D-4: Inline documentation is sparse.** Many public functions and types lack doc comments. The CONTRIBUTING.md recommends `# Arguments`, `# Returns`, `# Errors` sections, but few functions follow this convention.

### 3.13 Build & CI

**Finding B-1: Feature flags are well-designed.** `pdf` and `js-rendering` are optional features (both default). The slim build (`--no-default-features`) produces a smaller binary. This is good for deployment flexibility.

**Finding B-2: Release profile is optimized.** LTO, strip symbols, 1 codegen unit, panic=abort. Release builds should be performant.

**Finding B-3: CI scripts exist for CLI verification and legacy backend detection.** [scripts/ci/verify_cli.sh](../scripts/ci/verify_cli.sh) smoke-tests all subcommands. [scripts/ci/check_legacy_backend.sh](../scripts/ci/check_legacy_backend.sh) ensures no legacy backend references remain.

**Finding B-4: Cross-compilation for 5 targets.** The release workflow handles macOS (arm64/x86_64), Linux (x86_64/arm64), and Windows (x86_64) with both full and slim variants.

---

## 4. Comprehension Gap Analysis

Librarian is a **retrieval** system — it finds chunks of text semantically similar to a query. It is not yet a **comprehension** system. Here is what's missing to close that gap:

### 4.1 What Librarian Can Do Today

| Capability | Status | Quality |
|---|---|---|
| Index local files | ✅ | Good — gitignore, binary detection, incremental |
| Index web pages | ✅ | Good — SSRF, robots.txt, rate limiting, SPA support |
| Index sitemaps | ✅ | Good — recursive index parsing |
| Index images | ✅ | Good — relevance scoring, perceptual dedup |
| Index audio | ✅ | Fair — Whisper transcription, basic |
| Index video | ✅ | Fair — keyframe extraction, audio track |
| Chunk with structure awareness | ✅ | Good — heading hierarchy preserved |
| Vector similarity search | ✅ | Good — validated dimensions, over-fetch |
| BM25 hybrid search | ⚠️ | Weak — IDF=1.0 effectively disables it |
| Cross-encoder reranking | ✅ | Good — optional, partitions by text length |
| MCP search interface | ✅ | Good — clean JSON-RPC protocol |
| Background ingestion via MCP | ✅ | Good — async with fresh connections |

### 4.2 What a Comprehension System Needs

| Capability | Status | Impact |
|---|---|---|
| **Document relationships** | ❌ Missing | Cannot answer "what documents link to X?" or trace dependencies |
| **Section hierarchy navigation** | ❌ Missing | Cannot answer "what sections exist in this doc?" or "what's above this chunk?" |
| **Cross-reference resolution** | ❌ Missing | Links extracted during parsing but discarded — no graph built |
| **Query intent classification** | ❌ Missing | All queries treated identically regardless of user intent |
| **Multi-hop retrieval** | ❌ Missing | Cannot chain retrievals to answer complex questions spanning documents |
| **Concept extraction / tagging** | ❌ Missing | No automatic topic/concept tagging for faceted search |
| **Document summarization** | ❌ Missing | No MCP tool to get a document or section summary |
| **Temporal awareness** | ❌ Missing | No awareness of document versions, changelogs, or freshness |
| **Named vectors per modality** | ❌ Missing | Convention documented but not implemented; text and images share vector space |
| **Corpus-level BM25** | ❌ Missing | IDF statistics not computed across documents |
| **Payload-based Qdrant indexes** | ❌ Missing | No payload indexes for efficient metadata filtering |
| **Query expansion** | ❌ Missing | No synonym expansion or query reformulation |
| **Relevance feedback** | ❌ Missing | No mechanism to learn from which results users actually use |

### 4.3 Codanna Analogy

If we compare Librarian to Codanna (code intelligence tool): Codanna can find symbols, trace call graphs, analyze impact, and understand relationships between code entities. Librarian can find *text chunks* but cannot trace *knowledge dependencies*, navigate *document hierarchies*, or analyze the *impact of a documentation change* on related content. Librarian is roughly at the "grep search" level of code intelligence — powerful for exact-match retrieval but lacking the structural understanding that makes a tool truly comprehension-aware.

---

## 5. Prioritized Issue List

### P0 — Critical (should fix before next release)

| ID | Finding | Module | Description |
|---|---|---|---|
| P0-1 | CH-1 | chunk/ | Code block integrity declared but never enforced — `find_code_blocks()` is dead code and break points split code blocks |
| P0-2 | M-1 | mcp/ | Fresh embedder created per MCP search call — adds seconds of latency per query |
| P0-3 | Q-5 | store/ | Multi-value search filters silently ignored — `source_ids` with >1 value skips the filter entirely |

### P1 — High (address in next 1-2 releases)

| ID | Finding | Module | Description |
|---|---|---|---|
| P1-1 | P-1 | parse/ | HTML heading position uses `text.find()` — wrong for repeated headings, causes incorrect chunk-to-heading mapping |
| P1-2 | Q-1 | rank/ | BM25 IDF=1.0 makes hybrid search weaker than pure vector — either compute IDF or remove the feature claim |
| P1-3 | A-3 | store/ | Named vectors convention documented but not implemented — text and image share one vector space |
| P1-4 | P-2 | parse/ | Markdown images not extracted as media — multimodal ingestion broken for .md files |
| P1-5 | M-3 | mcp/ | Background operation status not queryable — `rag_status` shows overall status but not running operations |

### P2 — Medium (address as capacity allows)

| ID | Finding | Module | Description |
|---|---|---|---|
| P2-1 | T-2 | tests/ | Only 1 integration test — no end-to-end pipeline coverage |
| P2-2 | P-3 | parse/ | No frontmatter parsing — metadata from YAML/TOML frontmatter is lost |
| P2-3 | C-3 | config/ | No config file versioning — schema changes have no migration path |
| P2-4 | ER-3 | store/ | No retry logic for Qdrant operations — transient failures immediately propagate |
| P2-5 | D-4 | src/ | Sparse inline documentation — many public items lack doc comments |
| P2-6 | E-4 | embed/ | Sequential modality processing in `embed_unified` — should use `tokio::try_join!` |

### P3 — Enhancement (future features)

| ID | Finding | Module | Description |
|---|---|---|---|
| P3-1 | DB-2 | meta/ | No document relationship schema — prerequisite for comprehension features |
| P3-2 | Q-3 | commands/ | No query classification or routing — all queries treated identically |
| P3-3 | M-2 | mcp/ | No comprehension MCP tools — no summarize, explain, or relate capabilities |
| P3-4 | P-4 | parse/ | No RST, AsciiDoc, or Jupyter Notebook support |
| P3-5 | A-1 | — | No query analytics or relevance feedback loop |

---

## 6. Recommendations

### Immediate (v1.0.x patch candidates)

1. **Fix code block chunking (P0-1):** Wire `find_code_blocks()` into `find_break_points()` to prevent splitting code blocks. This is already 80% implemented — the boundary detection code exists, it just needs to be called.

2. **Cache the MCP embedder (P0-2):** Store the `Embedder` instance in the MCP server state and reuse it across `rag_search` calls. Only recreate on config change or error recovery.

3. **Fix multi-value filter construction (P0-3):** The `SearchFilter.to_qdrant_filter()` method should handle `source_ids.len() > 1` using `should` conditions (OR) rather than silently dropping the filter.

### Short-term (v1.1)

1. **Implement corpus BM25 or remove the hybrid claim:** Either maintain IDF statistics in SQLite (updated on ingestion) and use them during ranking, or remove the hybrid search feature until it's properly implemented. The current IDF=1.0 is misleading.

2. **Fix HTML heading position detection:** Use a DOM walker that tracks character offsets rather than `text.find()`. The heading text could be annotated with a unique marker during text extraction to avoid the first-occurrence problem.

3. **Implement named vectors for multimodal:** The convention is already documented. Create separate vector spaces for text and image modalities. This enables proper dimension-per-modality and modality-specific search.

### Medium-term (v1.2+)

1. **Add document link graph to SQLite:** A `document_links` table (`from_doc_id`, `to_doc_id`, `link_text`, `link_type`) populated during parsing would enable cross-reference queries. Links are already extracted — they just need to be stored.

2. **Add MCP background operation tracking:** Store running operation IDs in a shared state or SQLite table. Extend `rag_status` to report active operations with progress.

3. **Improve test coverage:** Add integration tests for:
   - Full ingest-query round trip (local dir)
   - URL ingestion with mocked HTTP server
   - Update and prune workflows
   - Config validation error paths

### Long-term (v2.0 — comprehension layer)

1. **Concept extraction pipeline:** During ingestion, run a lightweight NER/topic model to tag chunks with concepts. Store concepts in SQLite with chunk associations. This enables faceted search and concept-aware retrieval.

2. **Query classification and routing:** Classify incoming queries (factual, procedural, comparative, navigational) and route to appropriate retrieval strategies. This could use a small classification model or rule-based heuristics.

3. **Section-aware retrieval:** Extend chunk payloads with section path (e.g., "API Reference > Authentication > OAuth2") and support hierarchical filtering. Users should be able to search within a specific section or navigate up/down the document tree.

---

*End of audit report. This document is research-only and contains no code modifications.*
