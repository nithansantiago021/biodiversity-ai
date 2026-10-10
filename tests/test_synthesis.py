from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation
from app.knowledge.ingestion import create_document, create_document_chunks
from app.knowledge.embeddings import generate_chunk_embeddings
from app.knowledge.synthesis import generate_grounded_recommendation


def test_generate_grounded_recommendation_returns_structured_output():
    db = SessionLocal()
    document = None
    chunks = []
    obs = None

    try:
        # Create test document and chunk
        document = create_document(
            db,
            title="Soil Acidification & Organic Carbon Management",
            source="FAO",
            organization="Food and Agriculture Organization",
            document_type="report",
        )

        pages = [
            {
                "page_number": 15,
                "text": "Applying organic compost increases soil organic carbon and mitigates low pH stress in agricultural soils.",
            }
        ]

        chunks = create_document_chunks(
            db, document, pages, chunk_size=200, chunk_overlap=50
        )
        generate_chunk_embeddings(db, chunks)

        # Create test observation
        obs = EnvironmentalObservation(
            latitude=13.0827,
            longitude=80.2707,
            soil_ph=5.1,
            soil_organic_carbon=0.4,
            soil_moisture=11.0,
            land_use="agriculture",
            land_cover="cropland",
            species_richness=12.0,
            habitat_diversity=0.3,
            temperature=29.0,
            rainfall=800.0,
            pollution_index=0.1,
            deforestation_rate=0.07,
        )
        db.add(obs)
        db.commit()
        db.refresh(obs)

        # Execute synthesis
        result = generate_grounded_recommendation(db, obs)

        assert result["observation_id"] == obs.id
        assert "ecological_summary" in result
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

        first_rec = result["recommendations"][0]
        assert "title" in first_rec
        assert "description" in first_rec
        assert "time_horizon" in first_rec
        assert "citations" in first_rec
        assert "confidence" in first_rec
        assert "tradeoffs" in first_rec
        assert isinstance(first_rec["tradeoffs"], str)
        assert len(first_rec["citations"]) > 0

    finally:
        if obs:
            db.delete(obs)
        for chunk in chunks:
            db.delete(chunk)
        db.flush()
        if document:
            db.delete(document)
        db.commit()
        db.close()
