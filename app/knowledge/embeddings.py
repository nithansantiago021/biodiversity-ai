from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from app.models.db_models import DocumentChunk

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embedding(text: str) -> list[float]:
    """Generated a 384-dimensional vector for a single text string."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_chunk_embeddings(
    db: Session, chunks: list[DocumentChunk]
) -> list[DocumentChunk]:
    """Computes vector embeddings for a list of DocumentChunks and persists them to the DB."""
    if not chunks:
        return []

    texts = [chunk.chunk_text for chunk in chunks]
    model = get_embedding_model()

    embeddings = model.encode(texts, convert_to_numpy=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.tolist()

    db.commit()
    return chunks
