# GitHub Copilot Instructions for librarian

## Project Overview

librarian is a high-performance local RAG (Retrieval Augmented Generation) CLI tool and MCP server for indexing and querying documentation. It's written in Rust and uses SQLite for metadata, Qdrant for vector storage, and the Xinference embedding backend.

## Development Environment

### Prerequisites
- Rust 1.70+ (via rustup)
- Qdrant vector database
- SQLite
- Xinference embedding backend (managed automatically or provided externally)

### Building and Testing
```bash
# Build
cargo build --release

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run -- <command>

# Format code (required before committing)
cargo fmt

# Lint code (must pass with no warnings)
cargo clippy -- -D warnings
```

Default builds include PDF and JS rendering. To exclude them, use
`--no-default-features` (optionally re-adding a single feature).

## Coding Standards

### Rust Style
- Follow standard Rust formatting via `cargo fmt`
- All clippy warnings must be resolved (`-D warnings`)
- Use `tracing` macros for logging with structured fields
- Example: `info!(source_id = %source.id, docs = count, "Ingestion complete");`

### Error Handling
- Use the `Result` type from `error.rs`
- Propagate errors with `?` operator
- Provide meaningful error messages with context

### Async Programming
- Use async functions for I/O operations
- Runtime: `tokio` with full features
- Use `async-trait` for async trait methods

### Documentation
- Add doc comments for all public items
- Include `# Arguments`, `# Returns`, and `# Errors` sections
- Example:
  ```rust
  /// Process a single document and return chunk statistics.
  ///
  /// # Arguments
  /// * `doc` - The document to process
  /// * `config` - Chunking configuration
  ///
  /// # Returns
  /// Tuple of (chunks_created, chunks_updated)
  ///
  /// # Errors
  /// Returns error if document processing fails
  pub async fn process_document(...) -> Result<(i32, i32)>
  ```

## Repository-Specific Conventions

### CRITICAL: Read CONVENTIONS.md
**Always** consult `CONVENTIONS.md` for required conventions. Key rules include:

#### Ingestion Runs Must Record Operation
- Always invoke `cmd_ingest_dir|url|sitemap` with the appropriate `RunOperation` (`Ingest`, `Update`, `Reindex`)
- Use `interactive = false` for background/update/reindex flows
- Use `interactive = true` for CLI-driven ingestion

#### Chunk Writes Must Use Canonical Document IDs
- **CRITICAL**: Always use the `Document` returned by `MetaDb::upsert_document` when writing chunks
- Never reuse a freshly generated UUID when the doc already exists
- This prevents `FOREIGN KEY constraint failed` errors
- Example: `let doc = db.upsert_document(&doc).await?; process_chunks(... &doc, ...)`

#### MCP Triggers Run Asynchronously
- Background MCP triggers must spawn work with new `MetaDb::connect` / `QdrantStore::connect` instances
- Use cloned `Config` and reconnect in background tasks
- Example: `tokio::spawn(run_update_background(config.clone(), ...))`

#### Multimodal Indexing Is Capability-Gated
- Multimodal crawling requires registry-approved multimodal embedding models
- Late-interaction models (ColPali/ColQwen2) are not supported for ingestion
- Check `src/models.rs` for approved models

#### Image Chunks Carry Modality Metadata
- Set `Chunk.modality = "text"` for text chunks
- Set `Chunk.modality = "image"` (with `media_url`/`media_hash`) for image assets
- Use modality-aware helpers to prevent accidental deletion

#### Write Operations Must Use Validated Qdrant Connections
- Use `QdrantStore::connect_validated()` for operations that write to Qdrant
- Validates collection dimensions match embedding config
- Prevents silent data corruption

## Testing

### Unit Tests
- Add unit tests in the same file as the code
- Integration tests go in `tests/` directory
- Run specific tests: `cargo test test_name`
- Run with output: `cargo test -- --nocapture`

### Integration Tests
- Require a running Qdrant instance
- Start Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant`
- Run: `cargo test --test '*' -- --ignored`

### Coverage Goals
- Focus on parsing logic (HTML, Markdown, text)
- Chunking algorithm and boundary detection
- Database operations
- Search filter construction

## Project Structure

```
src/
├── main.rs              # CLI entry point
├── lib.rs               # Library exports
├── error.rs             # Error types
├── embedding_backend.rs # HTTP embedding client
├── config/              # Configuration management
├── commands/            # CLI commands (init, ingest, query, etc.)
├── crawl/               # Web crawling (robots.txt, rate limiting, sitemap)
├── parse/               # Content parsing (HTML, Markdown, text)
├── chunk/               # Structure-aware document chunking
├── embed/               # Embedding generation
├── store/               # Qdrant integration
├── meta/                # SQLite metadata storage
├── rank/                # Hybrid BM25 + vector ranking
├── rerank/              # Cross-encoder reranking
└── mcp/                 # MCP server (JSON-RPC over stdio)
```

## Dependencies

### Adding New Dependencies
- Only add dependencies when absolutely necessary
- Check for security vulnerabilities before adding
- Prefer well-maintained, popular crates
- Update `Cargo.toml` with minimal version constraints

### Key Dependencies
- `clap` - CLI argument parsing
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `sqlx` - SQLite with migrations
- `qdrant-client` - Vector database
- `tracing` - Structured logging
- `serde` + `serde_json` - Serialization

## Security Guidelines

### Never Hardcode Secrets
- Use environment variables for API keys and credentials
- Example: `qdrant_api_key_env = "QDRANT_API_KEY"`

### Input Validation
- Validate all external input (URLs, file paths, user input)
- Sanitize HTML content during parsing
- Respect robots.txt for web crawling
- Apply rate limiting for external requests

### Dependencies
- Regularly update dependencies for security patches
- Review dependency security advisories

## MCP Server Development

### Protocol
- JSON-RPC 2.0 over stdio
- Tool handlers in `src/mcp/tools.rs`
- Background operations must spawn async tasks with fresh connections

### Exposed Tools
- `rag_search` - Search the index
- `rag_sources` - List sources
- `rag_status` - Get status
- `rag_ingest_source` - Background ingestion
- `rag_update` - Background updates
- `rag_reindex` - Background reindexing

## Common Patterns

### Logging with Context
```rust
use tracing::{info, debug, error};

info!(source_id = %source.id, "Starting ingestion");
debug!(url = %url, "Fetching page");
error!(error = %e, "Failed to parse document");
```

### Database Operations
```rust
// Always use the returned document
let doc = db.upsert_document(&doc).await?;
// Use doc.id for chunks
db.insert_chunk(&chunk, doc.id).await?;
```

### Async Error Handling
```rust
async fn process() -> Result<()> {
    let data = fetch_data().await?;
    let result = process_data(data).await?;
    Ok(result)
}
```

## Commit Message Format

Use conventional commit messages:
```
feat(crawl): add sitemap index support
fix(query): handle empty result sets gracefully
docs(readme): add MCP configuration example
refactor(chunk): simplify boundary detection
```

## Additional Resources

- **CONTRIBUTING.md** - Comprehensive development guide
- **CONVENTIONS.md** - Required repository-specific conventions
- **AGENTS.md** - Instructions for AI agents using bd (beads) issue tracking
- **README.md** - User-facing documentation and features

## Issue Tracking

This project uses **bd** (beads) for issue tracking. See `AGENTS.md` for workflow details:
```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Critical Reminders

1. **Always check CONVENTIONS.md** before modifying ingestion, MCP, or database code
2. **Use canonical document IDs** from `upsert_document()` to avoid FK constraint errors
3. **Format and lint** before committing (`cargo fmt` + `cargo clippy`)
4. **Test thoroughly** with both unit and integration tests
5. **Document public APIs** with comprehensive doc comments
6. **Validate Qdrant connections** before write operations
7. **Use structured logging** with contextual fields
