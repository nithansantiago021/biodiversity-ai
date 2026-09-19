import os
from typing import Any, cast
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sqlalchemy.orm import Session
from app.models.db_models import DocumentChunk
from app.config import settings

_model = None


def get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    global _model
    if _model is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is missing.")

        _model = HuggingFaceEndpointEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            huggingfacehub_api_token=hf_token,
        )
    return _model


def generate_embedding(text: str) -> list[float]:
    """Generates a 384-dimensional vector for a single query text string via API."""
    model = get_embedding_model()
    # embed_query sends an HTTP request to Hugging Face and returns a list[float]
    return model.embed_query(text)


def generate_chunk_embeddings(
    db: Session, chunks: list[DocumentChunk]
) -> list[DocumentChunk]:
    """Computes vector embeddings for a list of DocumentChunks via API and persists them."""
    if not chunks:
        return []

    texts: list[str] = [str(chunk.chunk_text) for chunk in chunks]
    model = get_embedding_model()

    # embed_documents takes a list[str] and returns list[list[float]]
    embeddings = model.embed_documents(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = cast(Any, embedding)

    db.commit()
    return chunks
