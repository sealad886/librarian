# ragctl

A high-performance local RAG (Retrieval Augmented Generation) CLI tool and MCP server for indexing and querying documentation.

## Features

- **Local Document Indexing**: Index local directories with automatic gitignore support
- **Web Crawling**: Crawl web pages with robots.txt respect and rate limiting
- **Sitemap Support**: Parse sitemap.xml files for efficient URL discovery
- **Semantic Search**: Hybrid BM25 + vector similarity search
- **MCP Server**: Expose RAG tools via Model Context Protocol for VS Code integration
- **Incremental Updates**: Smart content hashing for efficient re-indexing
- **Structure-Aware Chunking**: Respect document structure (headings, code blocks)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ragctl CLI                              │
├─────────────────────────────────────────────────────────────┤
│  Commands: init | ingest | list | status | query | mcp | ... │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│  SQLite  │    │  Qdrant  │    │ FastEmbed│
│ Metadata │    │ Vectors  │    │ Embedder │
└──────────┘    └──────────┘    └──────────┘
```

## Installation

### Prerequisites

- **Rust** (1.70+): Install via [rustup](https://rustup.rs/)
- **Qdrant**: Vector database (see [Qdrant Installation](#qdrant-setup))

### Build from Source

```bash
git clone https://github.com/sealad886/librarian.git
cd librarian/ragctl
cargo build --release

# Binary will be at target/release/ragctl
```

### Qdrant Setup

The easiest way to run Qdrant is via Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Or install natively: https://qdrant.tech/documentation/guides/installation/

## Quick Start

```bash
# Initialize ragctl (creates config and database)
ragctl init

# Index a local directory
ragctl ingest dir ./docs --name "My Docs"

# Index a website
ragctl ingest url https://docs.rs/tokio/latest/tokio/ --name "Tokio Docs"

# Index from a sitemap
ragctl ingest sitemap https://example.com/sitemap.xml --name "Example Site"

# Search the index
ragctl query "how to use async/await"

# View indexed sources
ragctl list

# Check system status
ragctl status
```

## Commands

### `init`

Initialize ragctl configuration and database.

```bash
ragctl init [OPTIONS]

Options:
  -c, --config <PATH>  Custom config directory (default: ~/.ragctl)
  -f, --force          Overwrite existing configuration
```

### `ingest`

Add content to the RAG index.

#### Directory Ingestion

```bash
ragctl ingest dir <PATH> [OPTIONS]

Options:
  -n, --name <NAME>     Human-readable source name
  -e, --extensions      File extensions to include (default: all supported)
  --exclude <PATTERN>   Glob patterns to exclude
```

Supports: Markdown, HTML, plain text, code files. Respects `.gitignore`.

#### URL Ingestion

```bash
ragctl ingest url <URL> [OPTIONS]

Options:
  -n, --name <NAME>       Human-readable source name
  --max-pages <N>         Maximum pages to crawl (default: 100)
  --max-depth <N>         Maximum link depth (default: 3)
  --same-domain           Only crawl same domain (default: true)
```

Features: robots.txt respect, rate limiting, automatic link following.

#### Sitemap Ingestion

```bash
ragctl ingest sitemap <URL> [OPTIONS]

Options:
  -n, --name <NAME>     Human-readable source name
  --max-pages <N>       Maximum pages to fetch (default: from config)
```

Supports: sitemap.xml, sitemap index files, plain text URL lists.

### `query`

Search the RAG index.

```bash
ragctl query <QUERY> [OPTIONS]

Options:
  -k, --limit <N>        Number of results (default: 5)
  -s, --source <ID>      Filter by source ID
  --min-score <SCORE>    Minimum similarity (0-1, default: 0.5)
  --json                 Output as JSON
```

### `list`

List all indexed sources.

```bash
ragctl list [OPTIONS]

Options:
  --json                 Output as JSON
```

### `status`

Show system status and statistics.

```bash
ragctl status [OPTIONS]

Options:
  --json                 Output as JSON
```

### `prune`

Remove stale documents and orphaned data.

```bash
ragctl prune [OPTIONS]

Options:
  -s, --source <ID>      Only prune specific source
  --dry-run              Preview changes without deleting
  --orphans              Also remove orphaned Qdrant points
```

### `reindex`

Re-embed all documents (useful after model changes).

```bash
ragctl reindex [OPTIONS]

Options:
  -s, --source <ID>      Only reindex specific source
  --batch-size <N>       Embedding batch size (default: 32)
```

### `remove`

Remove a source and all its data.

```bash
ragctl remove <SOURCE_ID>
```

### `mcp`

Start the MCP server for VS Code integration.

```bash
ragctl mcp
```

The MCP server communicates via stdio and exposes:
- `rag_search`: Search the index
- `rag_sources`: List sources
- `rag_status`: Get status

## Configuration

Configuration is stored in `~/.ragctl/config.toml` (or custom path).

```toml
# Qdrant connection
qdrant_url = "http://localhost:6333"
collection_name = "ragctl"

# Embedding model
[embedding]
model = "BAAI/bge-small-en-v1.5"
dimensions = 384

# Chunking settings
[chunk]
max_chars = 1500
min_chars = 100
overlap_chars = 200
prefer_heading_boundaries = true

# Query settings
[query]
default_k = 5
min_score = 0.5
bm25_weight = 0.3

# Crawl settings
[crawl]
user_agent = "ragctl/0.1 (https://github.com/sealad886/librarian)"
timeout_secs = 30
max_pages = 100
max_depth = 3
respect_robots_txt = true
request_delay_ms = 500
```

## MCP Integration with VS Code

Add to your VS Code `settings.json`:

```json
{
  "mcp.servers": {
    "ragctl": {
      "command": "/path/to/ragctl",
      "args": ["mcp"]
    }
  }
}
```

Then use the tools via GitHub Copilot or other MCP clients:
- Search documentation with `rag_search`
- List available sources with `rag_sources`
- Check system health with `rag_status`

## Supported Formats

### Local Files
- Markdown (`.md`, `.mdx`)
- HTML (`.html`, `.htm`)
- Plain text (`.txt`, `.text`)
- ReStructuredText (`.rst`)
- Code files (for documentation comments)

### Web Content
- HTML pages
- Markdown pages
- Sitemap XML files

## Technical Details

### Embedding Model

Uses [FastEmbed](https://crates.io/crates/fastembed) with BAAI/bge-small-en-v1.5:
- 384 dimensions
- Optimized for retrieval tasks
- Runs locally (no API calls)

### Vector Database

[Qdrant](https://qdrant.tech/) provides:
- Efficient cosine similarity search
- Filtering by metadata
- Persistence and scalability

### Metadata Storage

SQLite database stores:
- Source registry
- Document metadata
- Chunk information with content hashes
- Ingestion run history

### Chunking Strategy

Structure-aware chunking:
1. Prefer breaking at headings
2. Fall back to paragraph boundaries
3. Respect sentence boundaries
4. Maintain configurable overlap

### Ranking

Hybrid ranking combines:
- **Vector similarity** (semantic meaning)
- **BM25** (keyword matching) - configurable weight

## Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# View ragctl status
ragctl status
```

### Slow Embedding

First run downloads the model (~50MB). Subsequent runs are fast.
Consider reducing batch size if memory-constrained:

```bash
ragctl reindex --batch-size 16
```

### Index Corruption

```bash
# Rebuild from scratch
ragctl prune --orphans
ragctl reindex
```

## Environment Variables

```bash
RAGCTL_CONFIG=/path/to/config.toml  # Custom config path
RAGCTL_LOG=debug                    # Log level (trace/debug/info/warn/error)
QDRANT_URL=http://host:6333         # Override Qdrant URL
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
