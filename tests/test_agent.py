from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation
from app.agent.workflow import biodiversity_agent


def test_langgraph_agent_execution_end_to_end():
    db = SessionLocal()
    obs = None

    try:
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

        initial_state = {
            "observation_id": obs.id,
            "observation": obs,
            "db": db,
            "generated_queries": [],
            "scientific_evidence": [],
            "recommendation_response": {},
            "validation_passed": False,
            "errors": [],
        }

        final_state = biodiversity_agent.invoke(initial_state)

        assert final_state["observation_id"] == obs.id
        assert len(final_state["generated_queries"]) > 0
        assert len(final_state["scientific_evidence"]) > 0
        assert "ecological_summary" in final_state["recommendation_response"]
        assert final_state["validation_passed"] is True
        assert len(final_state["errors"]) == 0

    finally:
        if obs:
            db.delete(obs)
            db.commit()
        db.close()