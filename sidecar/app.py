from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="librarian-embedding-backend")


class ModelSpec(BaseModel):
    id: str
    family: Optional[str] = None
    modalities: List[str]
    embedding_dim: int
    multivector: Optional[bool] = None
    supports_mrl: Optional[bool] = None
    max_batch: Optional[int] = None

    def supports_text(self) -> bool:
        return "text" in self.modalities or "image_text" in self.modalities

    def supports_image(self) -> bool:
        return "image" in self.modalities or "image_text" in self.modalities

    def supports_joint(self) -> bool:
        return "image_text" in self.modalities


class BackendCapabilities(BaseModel):
    backend_version: Optional[str] = None
    models: List[ModelSpec]


class ProbeRequest(BaseModel):
    model: str
    text: str
    image_base64: Optional[str] = None
    image_mime: Optional[str] = None


class ProbeResponse(BaseModel):
    id: str
    family: Optional[str] = None
    modalities: List[str]
    embedding_dim: Optional[int] = None
    multivector: Optional[bool] = None
    supports_mrl: Optional[bool] = None
    max_batch: Optional[int] = None
    text_embeddings: Optional[List[List[float]]] = None
    image_embeddings: Optional[List[List[float]]] = None
    joint_embeddings: Optional[List[List[float]]] = None


class EmbedTextRequest(BaseModel):
    model: str
    inputs: List[str]


class ImageTextInput(BaseModel):
    image_base64: str
    image_mime: Optional[str] = None
    text: Optional[str] = None


class EmbedImageTextRequest(BaseModel):
    model: str
    inputs: List[ImageTextInput]


class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]


def load_models() -> List[ModelSpec]:
    default_path = Path(__file__).with_name("models.json")
    path = Path(os.environ.get("LIBRARIAN_EMBEDDING_MODELS_PATH", default_path))
    if not path.exists():
        raise RuntimeError(f"Model registry not found: {path}")
    data = json.loads(path.read_text())
    return [ModelSpec(**item) for item in data]


MODEL_REGISTRY = load_models()
MODEL_INDEX = {model.id: model for model in MODEL_REGISTRY}
BACKEND_VERSION = os.environ.get("LIBRARIAN_EMBEDDING_BACKEND_VERSION")


def embed_vector(seed: bytes, dimension: int) -> List[float]:
    if dimension <= 0:
        raise HTTPException(status_code=400, detail="embedding_dim must be > 0")
    values: List[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
        for offset in range(0, len(digest), 4):
            if len(values) >= dimension:
                break
            chunk = digest[offset : offset + 4]
            number = int.from_bytes(chunk, "little")
            values.append((number / 2**32) * 2 - 1)
        counter += 1
    return values


def decode_image(image_base64: str) -> bytes:
    try:
        return base64.b64decode(image_base64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image_base64") from exc


def get_model(model_id: str) -> ModelSpec:
    model = MODEL_INDEX.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return model


def validate_batch(max_batch: Optional[int], count: int) -> None:
    if max_batch is not None and count > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {count} exceeds max_batch {max_batch}",
        )


@app.get("/capabilities", response_model=BackendCapabilities)
async def capabilities() -> BackendCapabilities:
    return BackendCapabilities(backend_version=BACKEND_VERSION, models=MODEL_REGISTRY)


@app.post("/probe", response_model=ProbeResponse)
async def probe(request: ProbeRequest) -> ProbeResponse:
    model = get_model(request.model)
    text_embedding: Optional[List[List[float]]] = None
    image_embedding: Optional[List[List[float]]] = None
    joint_embedding: Optional[List[List[float]]] = None

    if model.supports_text():
        text_embedding = [embed_vector(request.text.encode("utf-8"), model.embedding_dim)]

    if model.supports_image() and request.image_base64:
        image_bytes = decode_image(request.image_base64)
        if model.supports_joint():
            combined = request.text.encode("utf-8") + image_bytes
            joint_embedding = [embed_vector(combined, model.embedding_dim)]
        else:
            image_embedding = [embed_vector(image_bytes, model.embedding_dim)]
    elif model.supports_joint() and request.image_base64 is None:
        raise HTTPException(status_code=400, detail="image_base64 is required for joint models")

    return ProbeResponse(
        id=model.id,
        family=model.family,
        modalities=model.modalities,
        embedding_dim=model.embedding_dim,
        multivector=model.multivector,
        supports_mrl=model.supports_mrl,
        max_batch=model.max_batch,
        text_embeddings=text_embedding,
        image_embeddings=image_embedding,
        joint_embeddings=joint_embedding,
    )


@app.post("/v1/embed/text", response_model=EmbeddingsResponse)
async def embed_text(request: EmbedTextRequest) -> EmbeddingsResponse:
    model = get_model(request.model)
    if not model.supports_text():
        raise HTTPException(status_code=400, detail="Model does not support text inputs")
    validate_batch(model.max_batch, len(request.inputs))
    embeddings = [embed_vector(text.encode("utf-8"), model.embedding_dim) for text in request.inputs]
    return EmbeddingsResponse(embeddings=embeddings)


@app.post("/v1/embed/image_text", response_model=EmbeddingsResponse)
async def embed_image_text(request: EmbedImageTextRequest) -> EmbeddingsResponse:
    model = get_model(request.model)
    if not model.supports_image():
        raise HTTPException(status_code=400, detail="Model does not support image inputs")
    validate_batch(model.max_batch, len(request.inputs))
    embeddings: List[List[float]] = []
    for item in request.inputs:
        image_bytes = decode_image(item.image_base64)
        if model.supports_joint():
            text_bytes = (item.text or "").encode("utf-8")
            embeddings.append(embed_vector(text_bytes + image_bytes, model.embedding_dim))
        else:
            embeddings.append(embed_vector(image_bytes, model.embedding_dim))
    return EmbeddingsResponse(embeddings=embeddings)
