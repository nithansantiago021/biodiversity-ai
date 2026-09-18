from app.database import SessionLocal
from app.knowledge.ingestion import create_document, create_document_chunks


def test_document_chunks_have_overlap_and_rich_metadata():
    db = SessionLocal()
    document = None
    chunks = []

    try:
        document = create_document(
            db,
            title="Overlap Test Report",
            source="Test Source",
            organization="Test Organization",
            document_type="report",
        )

        pages = [
            {
                "page_number": 1,
                "text": (
                    "Biodiversity supports ecosystem functions and services. "
                    "Healthy soils contain diverse communities of microorganisms. "
                    "Microorganisms contribute to nutrient cycling and organic matter "
                    "decomposition. Soil biodiversity therefore supports ecosystem health."
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

        assert len(chunks) >= 2

        # Verify consecutive chunks share trailing context
        overlap_words = set(chunks[0].chunk_text.split()) & set(
            chunks[1].chunk_text.split()
        )
        assert overlap_words

        # Verify page provenance and rich metadata
        assert all(chunk.page_number == 1 for chunk in chunks)
        assert chunks[0].chunk_metadata["chunk_index"] == 0
        assert chunks[1].chunk_metadata["chunk_index"] == 1
        assert chunks[0].chunk_metadata["document_title"] == "Overlap Test Report"
        assert chunks[0].chunk_metadata["organization"] == "Test Organization"

    finally:
        for chunk in chunks:
            db.delete(chunk)

        db.flush()

        if document:
            db.delete(document)

        db.commit()
        db.close()
