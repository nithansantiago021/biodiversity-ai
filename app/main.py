from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation as EnvironmentalObservationDB
from app.models.schemas import EnvironmentalObservation
from app.knowledge.synthesis import generate_grounded_recommendation


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
    Triggers grounded scientific RAG synthesis for a stored EnvironmentalObservation ID.
    """
    # Fixed: Query the SQLAlchemy ORM model (EnvironmentalObservationDB), not the Pydantic schema
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

    recommendation_payload = generate_grounded_recommendation(db, obs)
    return recommendation_payload
