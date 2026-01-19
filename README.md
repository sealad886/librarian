# librarian

A high-performance local RAG (Retrieval Augmented Generation) CLI tool and MCP server for indexing and querying documentation.

## Features

- **Local Document Indexing**: Index local directories with automatic gitignore support
- **Web Crawling**: Crawl web pages with robots.txt respect and rate limiting
- **Sitemap Support**: Parse sitemap.xml files for efficient URL discovery
- **Semantic Search**: Hybrid BM25 + vector similarity search
- **MCP Server**: Expose RAG tools via Model Context Protocol for VS Code integration
- **Incremental Updates**: Smart content hashing for efficient re-indexing
- **Structure-Aware Chunking**: Respect document structure (headings, code blocks)
- **Safe Re-ingestion**: Re-running ingest/update reuses canonical document IDs to avoid FK issues and keeps chunk history consistent
- **Multimodal (Images)**: Optional image discovery, caching, and embeddings (config-gated)

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                      librarian CLI                           │
├──────────────────────────────────────────────────────────────┤
│  Commands: init | ingest | list | status | query | mcp | ... │
└──────────────────────┬───────────────────────────────────────┘
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
cd librarian/librarian
cargo build --release --all-features
# optionally add the executable to your PATH
#   [ ! -d /usr/local/bin ] && mkdir -p /usr/local/bin
#   ln -s -F "$(pwd)/target/release/librarian" /usr/local/bin/  # if symlinks are supported on your system
#   cp "$(pwd)/target/release/librarian" /usr/local/bin/  # to copy the executable directly

# Binary will be at target/release/librarian
```

### Qdrant Setup

The easiest way to run Qdrant is via Docker:

!!! Warning
    Running this command may remove and existing Qdrant Docker containers that are running if you previously used default settings

```bash
qdrant_for_librarian="qdrant_librarian"
docker run -p 6333:6333 -p 6334:6334 \
    -v ${HOME}/.librarian/qdrant_storage:/qdrant/storage:z \
    --name ${qdrant_for_librarian} \
    qdrant/qdrant -d
```

Or install natively: [Qdrant installation guide](https://qdrant.tech/documentation/guides/installation/)

## Quick Start

```bash
# Initialize librarian (creates config and database)
librarian init

# Index a local directory
librarian ingest dir ./docs --name "My Docs"

# Index a website
librarian ingest url https://docs.rs/tokio/latest/tokio/ --name "Tokio Docs"

# Index from a sitemap
librarian ingest sitemap https://example.com/sitemap.xml --name "Example Site"

# Search the index
librarian query "how to use async/await"

# View indexed sources
librarian list

# Check system status
librarian status

# What's New

- **Safe re-ingestion**: Chunk writes now always use the canonical document ID returned by SQLite, preventing `FOREIGN KEY constraint failed` errors during repeated ingest/update runs.
- **Better debugging**: Run with `RUST_LOG=debug` (or `-v` if using the CLI flag) to see which document IDs are used during ingestion for easier troubleshooting.
```

## Multimodal Indexing (Images)

Multimodal indexing is off by default. When enabled, librarian discovers image assets in HTML
(`img`, `picture`/`source`, and optional CSS backgrounds), caches them locally, and stores image
embeddings alongside text chunks.

To enable:

1. Set `embedding.model` to a supported multimodal embedding model (e.g., `Qwen/Qwen3-VL-Embedding-2B`, `Qwen/Qwen3-VL-Embedding-8B`, `jinaai/jina-clip-v2`, or `google/siglip2-*`).
2. Set `crawl.multimodal.enabled = true` and tune thresholds/limits.
3. Ensure your embedding dimensions are compatible with the configured backend.

Late-interaction models (e.g., `vidore/colpali`) are recognized but currently rejected for
multimodal ingestion. Image discovery filters out SVGs, applies size thresholds, and deduplicates
by URL plus perceptual hash. Cached assets are stored under `~/.librarian/assets`.

### Multimodal Model Support

Embedding models:

| Model | Strategy | Status |
| --- | --- | --- |
| `Qwen/Qwen3-VL-Embedding-2B` | VL embedding | Supported |
| `Qwen/Qwen3-VL-Embedding-8B` | VL embedding | Supported |
| `jinaai/jina-clip-v2` | Dual encoder | Supported |
| `google/siglip2-*` | Dual encoder | Supported (prefix match) |
| `vidore/colpali` | Late interaction | Recognized, ingestion blocked |

Reranker models:

| Model | Inputs | Status |
| --- | --- | --- |
| `Qwen/Qwen3-VL-Reranker-2B` | Text + image | Supported |
| `Qwen/Qwen3-VL-Reranker-8B` | Text + image | Supported |
| `jinaai/jina-reranker-m0` | Text + image | Supported |
| `lightonai/MonoQwen2-VL-v0.1` | Text + image | Supported |

References:

- Qwen3-VL embedding model cards: <https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B>, <https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B>
- Qwen3-VL reranker model cards: <https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B>, <https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B>
- Jina CLIP v2 model card: <https://huggingface.co/jinaai/jina-clip-v2>
- SigLIP2 model card (example): <https://huggingface.co/google/siglip2-base-patch16-224>
- ColPali model card: <https://huggingface.co/vidore/colpali>
- Jina reranker m0 model card: <https://huggingface.co/jinaai/jina-reranker-m0>
- MonoQwen2-VL model card: <https://huggingface.co/lightonai/MonoQwen2-VL-v0.1>

## Commands

### `init`

Initialize librarian configuration and database.

```bash
librarian init [OPTIONS]

Options:
  -c, --config <PATH>  Custom config directory (default: ~/.librarian)
  -f, --force          Overwrite existing configuration
  --non-interactive    Write defaults without prompts (for CI/scripting)
  -y, --yes            Accept defaults and skip confirmation
```

`librarian init` now runs an interactive configuration wizard. It writes a
`config.toml` that includes all settings, commenting out defaults and irrelevant
fields for easy discovery.

### `ingest`

Add content to the RAG index.

#### Directory Ingestion

```bash
librarian ingest dir <PATH> [OPTIONS]

Options:
  -n, --name <NAME>     Human-readable source name
  -e, --extensions      File extensions to include (default: all supported)
  --exclude <PATTERN>   Glob patterns to exclude
```

Supports: Markdown, HTML, plain text, code files. Respects `.gitignore`.

#### URL Ingestion

```bash
librarian ingest url <URL> [OPTIONS]

Options:
  -n, --name <NAME>       Human-readable source name
  --max-pages <N>         Maximum pages to crawl (default: 100)
  --max-depth <N>         Maximum link depth (default: 3)
  --same-domain           Only crawl same domain (default: true)
```

Features: robots.txt respect, rate limiting, automatic link following.

#### Sitemap Ingestion

```bash
librarian ingest sitemap <URL> [OPTIONS]

Options:
  -n, --name <NAME>     Human-readable source name
  --max-pages <N>       Maximum pages to fetch (default: from config)
```

Supports: sitemap.xml, sitemap index files, plain text URL lists.

### `query`

Search the RAG index.

```bash
librarian query <QUERY> [OPTIONS]

Options:
  -k, --limit <N>        Number of results (default: 5)
  -s, --source <ID>      Filter by source ID
  --min-score <SCORE>    Minimum similarity (0-1, default: 0.5)
  --json                 Output as JSON
```

### `list`

List all indexed sources.

```bash
librarian list [OPTIONS]

Options:
  --json                 Output as JSON
```

### `status`

Show system status and statistics.

```bash
librarian status [OPTIONS]

Options:
  --json                 Output as JSON
```

### `prune`

Remove stale documents and orphaned data.

```bash
librarian prune [OPTIONS]

Options:
  -s, --source <ID>      Only prune specific source
  --dry-run              Preview changes without deleting
  --orphans              Also remove orphaned Qdrant points
```

### `reindex`

Re-embed all documents (useful after model changes).

```bash
librarian reindex [OPTIONS]

Options:
  -s, --source <ID>      Only reindex specific source
  --batch-size <N>       Embedding batch size (default: 32)
```

### `remove`

Remove a source and all its data.

```bash
librarian remove <SOURCE_ID>
```

### `mcp`

Start the MCP server for VS Code integration.

```bash
librarian mcp
```

The MCP server communicates via stdio and exposes:

- `rag_search`: Search the index
- `rag_sources`: List sources
- `rag_status`: Get status

### `completions`

Generate shell completions for tab completion support.

```bash
# Bash (add to ~/.bashrc)
librarian completions bash >> ~/.bashrc

# Zsh (add to ~/.zshrc or create completion file)
librarian completions zsh > ~/.zsh/completions/_librarian

# Fish
librarian completions fish > ~/.config/fish/completions/librarian.fish

# PowerShell
librarian completions powershell >> $PROFILE
```

Supported shells: `bash`, `zsh`, `fish`, `powershell`, `elvish`

## Configuration

Configuration is stored in `~/.librarian/config.toml` (or custom path).
The init wizard writes a full config file, commenting out values that match
code defaults or are irrelevant to your selections.

```toml
# Qdrant connection
qdrant_url = "http://localhost:6333"
qdrant_api_key_env = "QDRANT_API_KEY"
collection_name = "librarian"

# Embedding model
[embedding]
model = "BAAI/bge-small-en-v1.5"
dimension = 384

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

# Optional reranker
[reranker]
enabled = false
model = "BAAI/bge-reranker-base"
top_k = 10

# Crawl settings
[crawl]
user_agent = "librarian/0.1 (https://github.com/sealad886/librarian)"
timeout_secs = 30
max_pages = 100
max_depth = 3
rate_limit_per_host = 2.0
respect_robots_txt = true
auto_js_rendering = true
js_page_load_timeout_ms = 30000
js_render_wait_ms = 2000
js_no_sandbox = false

# Multimodal crawling (images)
[crawl.multimodal]
enabled = false
include_images = true
include_audio = false
include_video = false
max_asset_bytes = 5000000
min_asset_bytes = 4096
max_assets_per_page = 10
allowed_mime_prefixes = ["image/"]
min_relevance_score = 0.6
include_css_background_images = false
```

## MCP Integration with VS Code

Add to your VS Code `settings.json`:

```json
{
  "mcp.servers": {
    "librarian": {
      "command": "/path/to/librarian",
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
- Image assets (optional, when multimodal is enabled)

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

# View librarian status
librarian status
```

### Slow Embedding

First run downloads the model (~50MB). Subsequent runs are fast.
Consider reducing batch size if memory-constrained:

```bash
librarian reindex --batch-size 16
```

### Index Corruption

```bash
# Rebuild from scratch
librarian prune --orphans
librarian reindex
```

## Environment Variables

```bash
LIBRARIAN_CONFIG=/path/to/config.toml  # Custom config path
LIBRARIAN_LOG=debug                    # Log level (trace/debug/info/warn/error)
QDRANT_URL=http://host:6333         # Override Qdrant URL
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
