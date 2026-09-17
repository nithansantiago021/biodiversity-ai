from datetime import date

from app.database import SessionLocal
from app.models.db_models import Document, DocumentChunk

def test_document_chunk_provenance():
    db = SessionLocal()

    try:
        document = Document(
            title="Soil Organic Carbon and Sustainable Agriculture",
            source="FAO",
            organization="Food and Agriculture Organization",
            publication_date=date(2021, 1, 1),
            url="https://example.com/fao-soil-carbon",
            document_type="report"
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        chunk = DocumentChunk(
            document_id=document.id,
            chunk_text=(
                "Soil organic carbon is an important indicator"
                "of soil health and ecosystem function."
            ),
            page_number=10,
            chunk_metadata={
                "topic": "soil_health",
                "indicator": "soil_organic_carbon"
            }
        )

        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        stored_chunk = db.query(DocumentChunk).filter(
            DocumentChunk.id == chunk.id
        ).first()

        assert stored_chunk is not None
        assert stored_chunk.document_id == document.id
        assert stored_chunk.chunk_text.startswith("Soil organic carbon")
        assert stored_chunk.chunk_metadata["topic"] == "soil_health"

        db.delete(stored_chunk)
        db.flush()
        db.delete(document)
        db.flush()
        db.commit()

    finally:
        db.close()