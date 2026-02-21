# Functional Deficits Report

This document outlines the functional deficits encountered during the audit of the `librarian` CLI tool.

## 1. Docker Dependency for Qdrant Initialization

**Command:** `librarian db init`

**Issue:** The command fails because the environment lacks a running Docker daemon.

**Details:**
The `librarian` tool relies on Qdrant as its vector database. In a local development or testing environment, Qdrant is typically run via Docker. Without Docker available, the `db init` command cannot establish a connection to a Qdrant instance, preventing the initialization of the vector collection.

**Impact:**
This blocks all write operations (ingestion, updates, reindexing) and any read operations (queries) that depend on the vector database.

**Proposed Fixes:**
- **Graceful Degradation:** Detect if Docker is running before attempting to connect to Qdrant. If Docker is not running, provide a clear, actionable error message instructing the user to start Docker or configure a remote Qdrant instance.
- **Embedded Vector Store Alternative:** Consider supporting an embedded vector database (like SQLite-VSS or an embedded Qdrant instance) for local development and testing to remove the hard dependency on Docker.
- **Configuration Check:** Add a `librarian doctor` or similar command to verify all external dependencies (Docker, Qdrant, Xinference) are available before running operations.

## 2. Xinference Model Download Blocking Execution

**Command:** `librarian ingest dir <PATH>` (and other ingest commands)

**Issue:** The command hangs indefinitely without providing feedback or failing gracefully.

**Details:**
Through codebase analysis, it was discovered that the CLI initializes the `Embedder` (which uses Xinference) *before* validating the Qdrant connection. Specifically, in `src/main.rs`, `create_embedder_auto()` is called before `QdrantStore::connect_validated()`.

When `create_embedder_auto()` is invoked, it attempts to ensure the required embedding model (e.g., `BAAI/bge-small-en-v1.5`) is launched via the local Xinference server. If the model is not already downloaded, Xinference begins downloading it synchronously.

The HTTP request to Xinference (`xinference_request_json` in `src/xinference/mod.rs`) has a timeout of 300 seconds (`XINFERENCE_HTTP_TIMEOUT_SECS`). During this time, the CLI appears to hang, providing no progress indication to the user about the model download.

Furthermore, because this initialization happens before the Qdrant connection check, the user experiences a long hang rather than an immediate failure indicating that the database is unavailable.

**Impact:**
- Poor user experience due to lack of progress feedback during initial model downloads.
- Misleading failure modes: users might think the tool is broken when it's actually downloading a large model, or they might wait a long time only to find out their database connection is misconfigured.

**Proposed Fixes:**
- **Initialization Order:** Refactor `src/main.rs` to validate the Qdrant connection (`QdrantStore::connect_validated()`) *before* initializing the `Embedder`. This will fail fast if the database is unavailable, improving the developer experience.
- **Download Progress:** Implement a mechanism to stream or poll download progress from Xinference, or at least log a clear message indicating that a model download is in progress and may take several minutes.
- **Timeout Handling:** Consider reducing the default HTTP timeout for Xinference requests or providing more granular timeouts for different types of operations (e.g., launching a model vs. embedding text).
