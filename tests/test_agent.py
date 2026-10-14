import uuid
from typing import cast
from langchain_core.runnables import RunnableConfig

from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation
from app.knowledge.ingestion import create_document, create_document_chunks
from app.knowledge.embeddings import generate_chunk_embeddings
from app.agent.workflow import biodiversity_agent, BiodiversityAgentState


def test_langgraph_agent_execution_end_to_end(knowledge_base):
    db = SessionLocal()
    obs = None
    doc = None
    chunks = []

    try:
        # 1. Seed a test document and chunk so the retrieval node finds evidence on empty DBs
        doc = create_document(
            db,
            title="Agent End-to-End Test Document",
            source="Agent Test Source",
            organization="Test Organization",
            document_type="report",
        )
        pages = [
            {
                "page_number": 1,
                "text": "Soil organic carbon increases water retention and supports soil microbial biodiversity in degraded agricultural lands.",
            }
        ]
        chunks = create_document_chunks(db, doc, pages)
        generate_chunk_embeddings(db, chunks)

        # 2. Seed observation
        obs = EnvironmentalObservation(
            latitude=13.0827,
            longitude=80.2707,
            soil_ph=5.2,
            soil_organic_carbon=0.4,
            soil_moisture=12.0,
            land_use="agriculture",
            land_cover="cropland",
            species_richness=14.0,
            habitat_diversity=0.3,
            temperature=28.5,
            rainfall=850.0,
            pollution_index=0.2,
            deforestation_rate=0.08,
        )
        db.add(obs)
        db.commit()
        db.refresh(obs)

        initial_state: BiodiversityAgentState = {
            "observation_id": cast(int, obs.id),
            "generated_queries": [],
            "scientific_evidence": [],
            "recommendation_response": {},
            "validation_passed": False,
            "errors": [],
        }

        # A checkpointed graph requires a thread_id in config.
        config: RunnableConfig = {
            "configurable": {"thread_id": f"test-agent-{uuid.uuid4()}"}
        }
        final_state = biodiversity_agent.invoke(initial_state, config=config)

        assert final_state["observation_id"] == obs.id
        assert len(final_state["generated_queries"]) > 0
        assert len(final_state["scientific_evidence"]) > 0
        assert "ecological_summary" in final_state["recommendation_response"]
        assert final_state["validation_passed"] is True
        assert len(final_state["errors"]) == 0

    finally:
        for chunk in chunks:
            db.delete(chunk)
        if doc:
            db.delete(doc)
        if obs:
            db.delete(obs)
        db.commit()
        db.close()
