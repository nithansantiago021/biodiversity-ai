from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation as EnvironmentalObservationDB
from app.models.schemas import EnvironmentalObservation
from app.agent.workflow import biodiversity_agent
from app.config import settings


app = FastAPI(
    title="Biodiversity AI",
    description="AI-powered environmental intelligence system",
    version="0.1.0"
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "message": "Biodiversity AI system online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "biodiversity-ai"
    }


@app.post("/observations", status_code=status.HTTP_201_CREATED)
def create_observation(
    observation: EnvironmentalObservation,
    db: Session = Depends(get_db),
):
    db_obs = EnvironmentalObservationDB(**observation.model_dump())
    db.add(db_obs)
    db.commit()
    db.refresh(db_obs)
    return {"id": db_obs.id, "message": "Environmental observation created successfully"}


@app.post("/observations/{observation_id}/recommendations")
def get_recommendation_for_observation(
    observation_id: int,
    db: Session = Depends(get_db),
):
    """
    Triggers the multi-step LangGraph Agent workflow for a stored EnvironmentalObservation ID.
    Executes grounding, two-stage vector retrieval, LLM synthesis, and provenance self-validation.
    """
    obs = (
        db.query(EnvironmentalObservationDB)
        .filter(EnvironmentalObservationDB.id == observation_id)
        .first()
    )
    if not obs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environmental observation with ID {observation_id} not found.",
        )

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

    if not final_state["validation_passed"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Agent provenance validation failed.",
                "errors": final_state["errors"],
            },
        )

    return final_state["recommendation_response"]
