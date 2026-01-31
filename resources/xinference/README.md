# Xinference Model Registry Snapshots

This directory contains generated, versioned snapshots of the Xinference model registry used by `librarian` for allowlisting and model capability checks.

## Files

- `registrations.embedding.json` — embedding model registrations
- `registrations.rerank.json` — reranker model registrations
- `registrations.audio.json` — audio model registrations
- `registrations.video.json` — video model registrations
- `registrations.image.json` — image model registrations
- `registrations.llm.json` — LLM model registrations
- `registrations.all.json` — merged view across all types
- `metadata.json` — snapshot hash + last-updated timestamp (stable unless data changes)

## Provenance

Snapshots are generated from a locally running Xinference instance. The sync tool calls Xinference’s OpenAPI-discovered registry endpoints and triggers per-type refresh when available (Model Update). The upstream source of truth is the Xinference Models Hub.

## Update Workflow

Run the sync generator:

- `cargo xtask xinference-sync --types embedding,rerank,audio,video --out resources/xinference --write`

Use `--dry-run` (default) to see diffs without writing.

To refresh a user-local cache instead of the embedded snapshots:

- `librarian xinference sync-models --endpoint http://127.0.0.1:9997 --write`
