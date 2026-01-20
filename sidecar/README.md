# Embedding Sidecar (HTTP)

This directory hosts a minimal HTTP embedding backend that implements the contract expected by librarian.
It is intended as a reference implementation and a lightweight local backend for development/testing.

## Endpoints

- `GET /capabilities`
- `POST /probe`
- `POST /v1/embed/text`
- `POST /v1/embed/image_text`

Responses follow the schema used by librarian's embedding backend client.

## Configuration

Model metadata is loaded from `models.json` by default. Override with:

- `LIBRARIAN_EMBEDDING_MODELS_PATH=/path/to/models.json`
- `LIBRARIAN_EMBEDDING_BACKEND_VERSION=...` (optional)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7997
```

## Notes

The sidecar returns deterministic embeddings derived from input content. Update `models.json` to align
dimensions and modalities with your backend.
