from app.database import SessionLocal
from app.knowledge.ingestion import create_document, create_document_chunks


def test_document_chunk_creation():
    db = SessionLocal()
    document = None
    chunks = []

    try:
        document = create_document(
            db,
            title="Test Biodiversity Report",
            source="Test Source",
            organization="Test Organization",
            document_type="report",
        )

        pages = [
            {
                "page_number": 1,
                "text": (
                    "Biodiversity is influenced by habitat diversity "
                    "and environmental conditions."
                ),
            },
            {
                "page_number": 2,
                "text": (
                    "Soil organic carbon can be used as an indicator "
                    "of soil health."
                ),
            },
        ]

        chunks = create_document_chunks(
            db,
            document,
            pages,
            chunk_size=100,
        )

        assert len(chunks) > 0

        assert all(
            chunk.document_id == document.id
            for chunk in chunks
        )

        assert chunks[0].page_number == 1

        assert chunks[-1].page_number == 2

        assert all(
            chunk.chunk_metadata["source_page"] == chunk.page_number
            for chunk in chunks
        )

    finally:
        # Safely clean up only what was successfully created
        for chunk in chunks:
            db.delete(chunk)
            db.flush()

        if document:
            db.delete(document)
            db.flush()

        db.commit()
        db.close()