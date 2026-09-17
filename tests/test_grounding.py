from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation
from app.knowledge.ingestion import create_document, create_document_chunks
from app.knowledge.embeddings import generate_chunk_embeddings
from app.knowledge.grounding import generate_observation_queries, analyze_observation_with_grounding


def test_generate_observation_queries_triggers_correctly():
    obs = EnvironmentalObservation(
        latitude=13.0827,
        longitude=80.2707,
        soil_ph=5.0,  # Acidic trigger
        soil_organic_carbon=0.5,  # Low SOC trigger
        soil_moisture=10.0,  # Low moisture trigger
        land_use="agriculture",
        land_cover="cropland",
        species_richness=10.0,
        habitat_diversity=0.2,  # Low habitat diversity trigger
        temperature=30.0,
        rainfall=500.0,
        pollution_index=0.1,
        deforestation_rate=0.09,  # High deforestation trigger
    )

    queries = generate_observation_queries(obs)

    # Asserts all 5 environmental triggers fired
    assert len(queries) == 5
    assert any("low soil organic carbon" in q for q in queries)
    assert any("acidic soil pH" in q for q in queries)
    assert any("soil moisture" in q for q in queries)
    assert any("habitat diversity" in q for q in queries)
    assert any("deforestation rate" in q for q in queries)


def test_analyze_observation_with_grounding_end_to_end():
    db = SessionLocal()
    document = None
    chunks = []
    obs = None

    try:
        document = create_document(
            db,
            title="Soil Health & Acidification Guide",
            source="UNEP",
            organization="United Nations Environment Programme",
            document_type="report",
        )

        pages = [
            {
                "page_number": 12,
                "text": "Acidic soils reduce microbial enzyme activity and restrict nutrient availability for crops.",
            }
        ]

        chunks = create_document_chunks(db, document, pages, chunk_size=200, chunk_overlap=50)
        generate_chunk_embeddings(db, chunks)

        obs = EnvironmentalObservation(
            latitude=10.0,
            longitude=75.0,
            soil_ph=4.8,
            soil_organic_carbon=2.0,
            soil_moisture=20.0,
            land_use="forestry",
            land_cover="tree_cover",
            species_richness=25.0,
            habitat_diversity=0.8,
            temperature=24.0,
            rainfall=1200.0,
            pollution_index=0.05,
            deforestation_rate=0.01,
        )
        db.add(obs)
        db.commit()
        db.refresh(obs)

        analysis = analyze_observation_with_grounding(db, obs, top_k_per_query=1)

        assert analysis["observation_id"] == obs.id
        assert len(analysis["scientific_evidence"]) > 0

        first_evidence = analysis["scientific_evidence"][0]
        assert "page_number" in first_evidence
        assert "provenance" in first_evidence
        assert "document_title" in first_evidence["provenance"]
        assert isinstance(first_evidence["page_number"], int)

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