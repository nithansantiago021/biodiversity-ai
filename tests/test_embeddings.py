from app.database import SessionLocal
from app.knowledge.ingestion import create_document, create_document_chunks
from app.knowledge.embeddings import generate_embedding, generate_chunk_embeddings


def test_generate_single_embedding():
    text = "Soil biodiversity contributes to ecosystem resilience and nutrient cycling."
    embedding = generate_embedding(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(val, float) for val in embedding)


def test_generate_chunk_embeddings_and_persistence():
    db = SessionLocal()
    document = None
    chunks = []

    try:
        document = create_document(
            db,
            title="Embedding Integration Test Report",
            source="Test Source",
            organization="Test Organization",
            document_type="report",
        )

        pages = [
            {
                "page_number": 1,
                "text": (
                    "Soil organic matter improves water retention capacity. "
                    "Microorganisms decompose organic inputs and release nutrients."
                ),
            }
        ]

        chunks = create_document_chunks(
            db,
            document,
            pages,
            chunk_size=100,
            chunk_overlap=20,
        )

        # Verify embeddings are initially None
        assert all(chunk.embedding is None for chunk in chunks)

        # Generate and persist embeddings
        updated_chunks = generate_chunk_embeddings(db, chunks)

        # Verify embeddings were attached and persisted
        assert len(updated_chunks) == len(chunks)
        for chunk in updated_chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 384

    finally:
        for chunk in chunks:
            db.delete(chunk)

        db.flush()

        if document:
            db.delete(document)

        db.commit()
        db.close()
