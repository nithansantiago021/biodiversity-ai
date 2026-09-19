import os
import requests
from typing import List, Tuple
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.models.db_models import DocumentChunk
from app.knowledge.embeddings import generate_embedding
from app.config import settings


# 1. Custom LangChain-compliant API Cross-Encoder
class HuggingFaceEndpointCrossEncoder(BaseModel):
    model_name: str = settings.RERANKER_MODEL_NAME
    hf_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Calls HF Inference API without downloading PyTorch into Render memory."""
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is missing.")

        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        scores = []
        for query, candidate in text_pairs:
            payload = {"inputs": {"source_sentence": query, "sentences": [candidate]}}
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=5)
                if res.status_code == 200:
                    scores.append(float(res.json()[0]))
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)

        return scores


# Lazy Singleton Instance
_reranker = None


def get_reranker() -> HuggingFaceEndpointCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = HuggingFaceEndpointCrossEncoder()
    return _reranker


# 2. Hybrid Retrieval & Reranking function
def search_similar_chunks(
    db: Session,
    query_text: str,
    top_k: int = 10,
    candidate_pool_size: int = 25,
) -> list[dict]:

    # STAGE 1: Candidate Generation via pgvector
    query_vector = generate_embedding(query_text)
    distance_expr = DocumentChunk.embedding.cosine_distance(query_vector)

    candidates = (
        db.query(DocumentChunk, distance_expr.label("distance"))
        .filter(DocumentChunk.embedding.is_not(None))
        .order_by(distance_expr)
        .limit(candidate_pool_size)
        .all()
    )

    if not candidates:
        return []

    # Prepare pairs for LangChain Cross-Encoder: [(query, chunk1), (query, chunk2)...]
    pairs = [(query_text, str(chunk.chunk_text)) for chunk, _ in candidates]

    # STAGE 2: Score using API Cross-Encoder
    reranker = get_reranker()
    rerank_scores = reranker.score(pairs)

    retrieved_chunks = []
    for (chunk, distance), score in zip(candidates, rerank_scores):
        retrieved_chunks.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_text": chunk.chunk_text,
                "rerank_score": round(float(score), 4),
                "vector_distance": round(float(distance), 4),
                "provenance": {
                    "document_title": chunk.document.title if chunk.document else None,
                    "organization": (
                        chunk.document.organization if chunk.document else None
                    ),
                    "source": chunk.document.source if chunk.document else None,
                    "chunk_metadata": chunk.chunk_metadata,
                },
            }
        )

    # Sort candidates by Cross-Encoder score descending
    retrieved_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return retrieved_chunks[:top_k]
