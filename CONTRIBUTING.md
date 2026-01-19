# Contributing to librarian

Thank you for your interest in contributing to librarian! This document provides guidelines and information for contributors.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Architecture Overview](#architecture-overview)

## Development Setup

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Qdrant (for integration testing)
- Docker (optional, for running Qdrant)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/sealad886/librarian.git
cd librarian/librarian

# Build in development mode
cargo build

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run -- status
```text

`librarian init` now runs an interactive configuration wizard. For scripted
setups, use `librarian init --non-interactive` to write defaults.

### Running Qdrant Locally

```bash
# Using Docker (recommended for development)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or use a persistent volume
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage:z \
    qdrant/qdrant
```

## Project Structure

```text
librarian/
├── Cargo.toml           # Project manifest
├── README.md            # User documentation
├── CONTRIBUTING.md      # This file
└── src/
    ├── main.rs          # CLI entry point and argument parsing
    ├── lib.rs           # Library exports
    ├── error.rs         # Error types and Result alias
    ├── config/          # Configuration management
    │   ├── mod.rs       # Config struct and loading
    │   └── defaults.rs  # Default configuration values
    ├── commands/        # CLI command implementations
    │   ├── mod.rs       # Re-exports
    │   ├── init.rs      # librarian init
    │   ├── ingest.rs    # librarian ingest (dir/url/sitemap)
    │   ├── query.rs     # librarian query
    │   ├── status.rs    # librarian status, list
    │   ├── prune.rs     # librarian prune, remove
    │   └── reindex.rs   # librarian reindex
    ├── crawl/           # Web crawling
    │   ├── mod.rs       # Crawler implementation
    │   ├── robots.rs    # robots.txt parsing
    │   ├── rate_limit.rs # Per-host rate limiting
    │   └── sitemap.rs   # Sitemap XML parsing
    ├── parse/           # Content parsing
    │   ├── mod.rs       # Content type detection, dispatcher
    │   ├── html.rs      # HTML extraction
    │   ├── markdown.rs  # Markdown processing
    │   └── text.rs      # Plain text handling
    ├── chunk/           # Document chunking
    │   ├── mod.rs       # Chunking algorithm
    │   └── boundaries.rs # Break point detection
    ├── embed/           # Embedding generation
    │   ├── mod.rs       # Embedder trait
    │   └── fastembed_impl.rs # FastEmbed implementation
    ├── store/           # Qdrant integration
    │   ├── mod.rs       # QdrantStore wrapper
    │   └── payload.rs   # Chunk payload types
    ├── meta/            # SQLite metadata
    │   ├── mod.rs       # MetaDb implementation
    │   └── schema.rs    # Database schema
    ├── rank/            # Result ranking
    │   └── mod.rs       # Hybrid BM25 + vector ranking
   ├── rerank/          # Cross-encoder reranking
   │   ├── mod.rs       # Reranker trait
   │   └── fastembed_impl.rs # FastEmbed reranker
    └── mcp/             # MCP server
        ├── mod.rs       # Module exports
        ├── server.rs    # stdio server loop
        ├── tools.rs     # Tool implementations
        └── types.rs     # MCP protocol types
```

## Code Style

### Formatting

We use standard Rust formatting:

```bash
# Format code
cargo fmt

# Check formatting (CI will fail if not formatted)
cargo fmt --check
```

### Linting

```bash
# Run clippy
cargo clippy -- -D warnings

# Run with all features
cargo clippy --all-features -- -D warnings
```

### Guidelines

1. **Error Handling**: Use the `Result` type from `error.rs`. Propagate errors with `?` operator.

2. **Logging**: Use `tracing` macros (`info!`, `debug!`, `warn!`, `error!`). Include structured fields:

   ```rust
   info!(source_id = %source.id, docs = count, "Ingestion complete");
   ```

3. **Ingestion invariants**: `MetaDb::upsert_document` returns the canonical document row for a `(source_id, uri)` pair. Always use the returned `Document` (and its `id`) for any chunk writes to avoid FK violations. See `meta::tests::test_reingest_document_uses_canonical_id_for_chunks` for a regression guard.

4. **Documentation**: Add doc comments for public items:

   ```rust
   /// Process a single document and return chunk statistics.
   ///
   /// # Arguments
   /// * `doc` - The document to process
   /// * `config` - Chunking configuration
   ///
   /// # Returns
   /// Tuple of (chunks_created, chunks_updated)
   pub async fn process_document(...) -> Result<(i32, i32)>
   ```

5. **Async**: Prefer async functions for I/O operations. Use `tokio` for the async runtime.

6. **Tests**: Add unit tests in the same file, integration tests in `tests/`.

## Testing

### Unit Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_chunk_document

# Run with output
cargo test -- --nocapture
```

### Integration Tests

Integration tests require a running Qdrant instance:

```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Run integration tests
cargo test --test '*' -- --ignored
```

### Test Coverage

We aim for good coverage of:

- Parsing logic (HTML, Markdown, text)
- Chunking algorithm and boundary detection
- Database operations
- Search filter construction

## Pull Request Process

1. **Fork and Branch**: Create a feature branch from `main`:

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make Changes**: Implement your feature or fix.

3. **Test**: Ensure all tests pass:

   ```bash
   cargo test
   cargo fmt --check
   cargo clippy -- -D warnings
   ```

4. **Commit**: Use conventional commit messages:

   ```text
   feat(crawl): add sitemap index support
   fix(query): handle empty result sets gracefully
   docs(readme): add MCP configuration example
   refactor(chunk): simplify boundary detection
   ```

5. **Push and PR**: Push your branch and open a pull request.

6. **Review**: Address any feedback from maintainers.

## Architecture Overview

### Data Flow

```text
User Input → Parse → Chunk → Embed → Store (Qdrant + SQLite)
                                          ↓
Query → Embed → Search (Qdrant) → Rank → Return Results
```

### Key Components

#### Config (`config/`)

- Loads from TOML file
- Provides defaults for all settings
- Validates configuration on load

#### Commands (`commands/`)

- Each command is a public async function
- Commands orchestrate other modules
- Handle user-facing output formatting

#### Crawl (`crawl/`)

- `Crawler`: Main crawling logic with BFS
- `RobotsRules`: Parse and check robots.txt
- `HostRateLimiter`: Per-host request throttling
- `SitemapParser`: Parse sitemap.xml files

#### Parse (`parse/`)

- Content type detection (extension + MIME)
- HTML: Uses `scraper` + `html2text`
- Markdown: Uses `pulldown-cmark`
- Extracts headings, code blocks, links

#### Chunk (`chunk/`)

- Structure-aware chunking
- Priority-based break point selection
- Configurable size and overlap
- Blake3 hashing for deduplication

#### Embed (`embed/`)

- `Embedder` trait for abstraction
- `FastEmbedder`: Local ONNX embedding
- Batch processing support
- Optional image embeddings when multimodal is enabled

#### Store (`store/`)

- `QdrantStore`: Qdrant client wrapper
- Point upsert/delete/search
- Filter construction from options

#### Meta (`meta/`)

- `MetaDb`: SQLite wrapper
- Sources, documents, chunks tables
- Ingestion run tracking

#### Rank (`rank/`)

- `Ranker`: Result scoring
- Hybrid BM25 + vector scoring
- Deduplication and filtering

#### Rerank (`rerank/`)

- `Reranker` trait for cross-encoder reranking
- `FastEmbedReranker` for local reranking
- Multimodal gating inferred from the model registry (`src/models.rs`)

#### MCP (`mcp/`)

- JSON-RPC 2.0 over stdio
- Tool definitions and handlers
- Integrates with query/status commands

### Design Principles

1. **Modularity**: Each module has a clear responsibility
2. **Testability**: Business logic separated from I/O
3. **Incrementality**: Content hashing enables efficient updates
4. **Resilience**: Graceful handling of failures (bad URLs, parse errors)
5. **Observability**: Structured logging throughout

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Include reproduction steps for bugs
- For questions, use GitHub Discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
