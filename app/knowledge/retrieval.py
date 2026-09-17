from sentence_transformers import CrossEncoder
from sqlalchemy.orm import Session
from app.models.db_models import DocumentChunk
from app.knowledge.embeddings import generate_embedding

# Lazy-load Cross-Encoder reranker
_reranker = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def search_similar_chunks(
    db: Session,
    query_text: str,
    top_k: int = 5,
    candidate_pool_size: int = 25,
) -> list[dict]:
    """
    Two-Stage Hybrid & Rerank Retrieval:
    Stage 1: Candidate Generation via pgvector Cosine Similarity.
    Stage 2: Cross-Encoder Reranking for deep query-chunk semantic alignment.
    """
    # STAGE 1: Fast candidate retrieval via pgvector
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

    # Prepare (Query, Chunk_Text) pairs for Cross-Encoder
    reranker = get_reranker()
    pairs = [[query_text, chunk.chunk_text] for chunk, _ in candidates]
    
    # STAGE 2: Deep joint-attention reranking
    rerank_scores = reranker.predict(pairs)

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
                    "organization": chunk.document.organization if chunk.document else None,
                    "source": chunk.document.source if chunk.document else None,
                    "chunk_metadata": chunk.chunk_metadata,
                },
            }
        )

    # Sort candidates by Cross-Encoder score descending
    retrieved_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return retrieved_chunks[:top_k]