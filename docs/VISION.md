# Librarian — Vision & Direction

## What Librarian Is

Librarian is a high-performance local RAG (Retrieval Augmented Generation) tool
that does for documentation comprehension what
[Codanna](https://github.com/bartolli/codanna) does for codebase understanding.

Where Codanna provides symbol-level search, call-graph tracing, impact analysis,
and semantic search with context for code intelligence — Librarian provides
document-level search, cross-reference awareness, section hierarchy navigation,
and comprehension-quality retrieval for documentation understanding.

It is designed to be:

- **Cheap to update** — incremental ingestion with content hashing
- **Fast to search** — vector similarity + hybrid BM25 ranking + optional reranking
- **Comprehension-first** — understanding over rote lookup

## What Librarian Does Today (v1.0)

- **Multi-source ingestion**: local directories, web pages, sitemaps
- **Structure-aware chunking**: respects headings, code blocks, paragraph breaks
- **Hybrid search**: Qdrant vector similarity combined with BM25 keyword scoring
- **Cross-encoder reranking**: optional reranker pass for precision retrieval
- **Multimodal indexing**: images (with embedding), audio (via transcription),
  video (via keyframe extraction) — all capability-gated
- **MCP server**: exposes `rag_search`, `rag_sources`, `rag_status`,
  `rag_ingest_source`, `rag_update`, `rag_reindex` over stdio JSON-RPC
- **Incremental updates**: content-hash diffing so only changed documents are
  re-processed
- **Safe re-ingestion**: canonical document IDs prevent FK constraint failures
- **Validated writes**: Qdrant dimension consistency checked before every write
- **Xinference management**: auto-launches and manages local embedding backend

## The Codanna Analogy

| Codanna (Code Intelligence)       | Librarian (Documentation Intelligence)             |
| --------------------------------- | -------------------------------------------------- |
| Find symbols by name / kind       | Find document chunks by content                    |
| Trace call graphs                 | Trace document cross-references (planned)          |
| Analyze change impact             | Analyze documentation dependency impact (planned)  |
| Semantic search with context      | Semantic search with heading / section context      |
| Navigate code structure           | Navigate document hierarchy (planned)              |

## The Comprehension Gap

Current retrieval returns the *most similar* chunks, but comprehension requires
more: awareness of how documents relate to each other, which sections are
prerequisites for others, and how concepts flow across a corpus. Closing this
gap is the long-term goal of Librarian.

Key gaps today:

1. **No cross-document linking** — search finds individual chunks but cannot
   follow references between documents.
2. **No query intent classification** — all queries are treated as single-hop
   similarity lookups, even when multi-step reasoning is needed.
3. **No temporal awareness** — all documents are treated equally regardless of
   freshness or version.
4. **No concept extraction** — the system stores text and vectors, but does not
   model high-level topics or concepts.

## Roadmap

### v1.1 — Foundation Hardening

- Code block chunking integrity ✅ (done)
- Cached MCP embedder ✅ (done)
- Multi-value search filters ✅ (done)
- HTML heading accuracy ✅ (done)
- Real BM25 IDF scoring ✅ (done)
- Markdown image + frontmatter extraction ✅ (done)

### v1.2 — Retrieval Intelligence

- Document link graph in SQLite
- Named vectors per modality
- Payload-based Qdrant indexes
- Background operation tracking
- Query expansion

### v2.0 — Comprehension Layer

- Document relationship tracking & cross-reference navigation
- Section hierarchy navigation
- Query intent classification & routing
- Multi-hop retrieval for complex questions
- Concept extraction & tagging
- Document summarization MCP tools
- Temporal awareness (versioning, freshness)

## Design Principles

1. **Local-first**: All data stays on the user's machine.
2. **Incremental**: Updates are cheap — only changed content is re-processed.
3. **Modality-aware**: Text, images, audio, video handled distinctly.
4. **Fail-fast**: Configuration errors caught before ingestion begins.
5. **Observable**: Structured logging, status reporting, progress tracking.
6. **Extensible**: Clean module boundaries, trait-based abstractions.

## How to Contribute

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup, code style, and
testing guidance. The canonical conventions are in
[CONVENTIONS.md](../CONVENTIONS.md).
