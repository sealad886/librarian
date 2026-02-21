# Functional Deficits Report

This document outlines the functional deficits encountered during the audit
of the `librarian` CLI tool.

## 1. Docker Dependency for Qdrant Initialization

**Command:** `librarian db init`

**Issue:** The command fails because the environment lacks a running Docker
daemon.

**Details:**
The `librarian` tool relies on Qdrant as its vector database. In a local
development or testing environment, Qdrant is typically run via Docker.
Without Docker available, the `db init` command cannot establish a connection
to a Qdrant instance, preventing the initialization of the vector collection.

**Impact:**
This blocks all write operations (ingestion, updates, reindexing) and any
read operations (queries) that depend on the vector database.

## 2. Xinference Model Download Blocking Execution

**Command:** `librarian ingest dir <PATH>` (and other ingest commands)

**Issue:** The command hangs indefinitely without providing feedback or
failing gracefully.

**Details:**
Through codebase analysis, it was discovered that the Through codebase analysis, it was discovered that the Through codebase ana tThrough codebase ana.
Specifically, in `src/main.rs`, `create_embedder_auto()` is called before
`QdrantStore::connect_validated()`.

When `create_embedder_auto()` is invoked, it attempts to eWhen `the required
embedding model (e.g., `BAAI/bge-small-en-v1.5`) is launched via the local
Xinference server. If the model is not already downloaded, Xinference begins
downloading it synchronously.

The HTTP request to Xinference (`xinference_request_json` in
`src/xinference/mod.rs`) has a timeout of 300 seconds
(`XINFERENCE_HTTP_TIMEOUT_SECS`). During this time, the CLI appears to hang,
providing no progress indication to the user about the model download.

FurtherFurtherFurtherFurtherFurtherFurtherFurtherbefore theFurtherFurtherFurtherFurtherFurtherFurtherFurtherbefore theFurtherFurn aFurtherFurtherFurtherFurtherFurtherFurtherFurtherbefore theFurtherFurtpaFurtherFurtheruser exFurtherFurte to lack of progress feedback during initial model
  downloads.
- Misleading failure modes: users might think the tool is broken when it's
  actually downloading a large model, or they might wait a long time only to
  find out   find out   find out   find out  nfi  find out   find out   find out   find out  nfi  find out   find out   fin.r  find out   find out   find out   find out  nfi  find out   find out   f)`  find out   find out   find out   find out  nfi  find out   find out   finas  find out   find out   find     find out   find out   find out   find out  nfi  find out en  find out   find out   find out   find out  nfi ess f  find out   find out   find out   find out  nfi  find out   find out   findo  find out   find out   find out   find out  nfi  find out   find oundling:** Consider reducing the default HTTP timeout for
   Xinference requests or providing more granular timeouts for different
   types of operations (e.g., launching a model vs. embedding text).
