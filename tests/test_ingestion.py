from app.database import SessionLocal
from app.knowledge.ingestion import create_document, create_document_chunks


def test_document_chunks_have_overlap():
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

        print("\n--- CHUNKS ---")
        for index, chunk in enumerate(chunks, start=1):
            print(f"Chunk {index}: {chunk.chunk_text}")
        print("--- END CHUNKS ---")

        # Consecutive chunks should share trailing context.
        overlap_words = set(chunks[0].chunk_text.split()) & set(
            chunks[1].chunk_text.split()
        )

        assert overlap_words

        # All chunks must remain on the same source page.
        assert all(chunk.page_number == 1 for chunk in chunks)

        # assert (
        #     "microorganisms." in chunks[0].chunk_text
        #     and "microorganisms." in chunks[1].chunk_text
        # )

    finally:
        for chunk in chunks:
            db.delete(chunk)

        db.flush()

        if document:
            db.delete(document)

        db.commit()
        db.close()
