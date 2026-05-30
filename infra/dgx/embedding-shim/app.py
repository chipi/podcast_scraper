"""FastAPI embedding shim for DGX (RFC-089).

Exposes POST /embed with the same model id as local sentence-transformers indexing.
GPU vectors are not bit-identical to CPU; consumers assert functional top-K equivalence.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="podcast-scraper-dgx-embedding-shim", version="1.0.0")

_model = None
_model_id: Optional[str] = None


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1)
    model: str = DEFAULT_MODEL
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int


def _load_model(model_id: str):
    global _model, _model_id
    if _model is not None and _model_id == model_id:
        return _model
    from sentence_transformers import SentenceTransformer

    logger.info("loading embedding model %s", model_id)
    _model = SentenceTransformer(model_id)
    _model_id = model_id
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(body: EmbedRequest) -> EmbedResponse:
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts required")
    try:
        model = _load_model(body.model)
        vectors = model.encode(
            body.texts,
            normalize_embeddings=body.normalize,
            batch_size=64,
        )
    except Exception as exc:
        logger.exception("embed failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if hasattr(vectors, "tolist"):
        rows = vectors.tolist()
    else:
        rows = [list(v) for v in vectors]
    dim = len(rows[0]) if rows else 0
    return EmbedResponse(embeddings=rows, model=body.model, dim=dim)
