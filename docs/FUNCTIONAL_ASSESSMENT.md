# Functional Assessment

This document outlines the CLI commands, subcommands, options, and expected outcomes for the `librarian` tool.

## Global Options

These options can be applied to any command:

- `-c, --config <CONFIG>`: Path to config file
- `-v, --verbose`: Enable verbose logging
- `--json`: Output as JSON
- `-h, --help`: Print help
- `-V, --version`: Print version

## Commands

### `init`

Initialize librarian configuration and database.

**Options:**
- `--force`: Force overwrite existing config
- `--non-interactive`: Run without interactive prompts (writes defaults)
- `-y, --yes`: Accept defaults and skip confirmation

**Expected Outcome:**
Creates a new configuration file (`config.toml`) and initializes the SQLite metadata database. If run interactively, prompts the user for configuration values.

### `ingest`

Ingest documentation into the RAG index.

#### `ingest dir`

Ingest a local directory.

**Options:**
- `<PATH>`: Path to directory
- `-n, --name <NAME>`: Source name (defaults to directory name)
- `--extensions <EXTENSIONS>`: File extensions to include (e.g., md,txt,html)
- `--exclude <EXCLUDE>`: Exclude patterns (glob)

**Expected Outcome:**
Reads files from the specified directory, chunks them, generates embeddings, and stores them in Qdrant. Updates the metadata database with source and document information.

#### `ingest url`

Ingest a URL (with crawling).

**Options:**
- `<URL>`: URL to ingest
- `-n, --name <NAME>`: Source name (defaults to domain)
- `--max-pages <MAX_PAGES>`: Maximum pages to crawl [default: 100]
- `--max-depth <MAX_DEPTH>`: Maximum crawl depth [default: 3]
- `--path-prefix <PATH_PREFIX>`: Restrict crawling to this path prefix

**Expected Outcome:**
Crawls the specified URL up to the given depth and page limit, extracts content, chunks it, generates embeddings, and stores them in Qdrant.

#### `ingest sitemap`

Ingest URLs from a sitemap.

**Options:**
- `<URL>`: Sitemap URL
- `-n, --name <NAME>`: Source name
- `--max-pages <MAX_PAGES>`: Maximum pages to fetch

**Expected Outcome:**
Fetches the sitemap, extracts URLs, crawls them, and processes the content into the RAG index.

### `query`

Query the RAG index.

**Options:**
- `<QUERY>`: The search query
- `-l, --limit <LIMIT>`: Maximum number of results [default: 5]
- `-m, --min-score <MIN_SCORE>`: Minimum similarity score (0-1)
- `--source <SOURCE>`: Filter to specific source IDs
- `--dedupe`: Deduplicate results by document

**Expected Outcome:**
Generates an embedding for the query, searches Qdrant for similar chunks, and returns the top results with their content and metadata.

### `status`

Show system status.

**Expected Outcome:**
Displays information about the configuration, database, Qdrant connection, and overall system health.

### `sources`

List registered sources.

**Options:**
- `--ids-only`: Output only source IDs (one per line, for scripting)

**Expected Outcome:**
Lists all sources currently tracked in the metadata database, including their IDs, types, URIs, and document counts.

### `prune`

Remove stale documents and orphan points.

**Options:**
- `--dry-run`: Dry run - show what would be removed
- `--remove-orphans`: Also remove orphan Qdrant points
- `--source <SOURCE>`: Only prune specific source IDs

**Expected Outcome:**
Identifies and removes documents from the database that no longer exist in their original sources, and optionally cleans up orphaned vectors in Qdrant.

### `reindex`

Re-embed all documents.

**Options:**
- `--source <SOURCE>`: Only reindex specific source IDs
- `--batch-size <BATCH_SIZE>`: Batch size for embedding [default: 32]

**Expected Outcome:**
Regenerates embeddings for all existing chunks in the database and updates them in Qdrant. Useful when changing embedding models.

### `update`

Incrementally update sources and prune embeddings.

**Options:**
- `--source <SOURCE>`: Only update specific source IDs
- `--skip-prune`: Skip pruning orphan vectors after updating

**Expected Outcome:**
Re-runs ingestion for existing sources, only processing new or modified documents, and then prunes stale documents.

### `remove`

Remove a source and all its data.

**Options:**
- `<SOURCE_ID>`: Source ID to remove

**Expected Outcome:**
Deletes the specified source, all its documents, chunks, and associated Qdrant vectors.

### `rename`

Rename an existing source.

**Options:**
- `<SOURCE_ID>`: Source ID to rename
- `<NAME>`: New name to set

**Expected Outcome:**
Updates the display name of the specified source in the metadata database.

### `mcp`

Start MCP server on stdio.

**Expected Outcome:**
Starts the Model Context Protocol server, listening for JSON-RPC requests on standard input/output.

### `completions`

Generate shell completions.

**Options:**
- `<SHELL>`: Shell to generate completions for (bash, elvish, fish, powershell, zsh)

**Expected Outcome:**
Outputs shell completion script for the specified shell.

### `db`

Manage Qdrant vector database.

#### `db init`
Initialize/create the Qdrant collection.

#### `db status`
Show Qdrant collection status.

#### `db check`
Check collection health and configuration consistency.

#### `db reset`
Reset the collection (delete all vectors and recreate).
**Options:** `--yes` (Skip confirmation prompt)

### `config`

Configuration management commands.

#### `config print`
Print default configuration with all comments.
**Options:** `--all` (Print all options including defaults)

### `xinference`

Xinference registry tools.

#### `xinference sync-models`
Sync Xinference model registry snapshots to the local cache.
**Options:** `--endpoint`, `--types`, `--write`, `--skip-update`, `--cache-dir`, `--retries`, `--timeout-secs`
