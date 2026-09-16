from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation as EnvironmentalObservationDB
from app.models.schemas import EnvironmentalObservation


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


@app.post("/observations")
def create_observation(
    observation: EnvironmentalObservation,
    db: Session = Depends(get_db)
):

    db_observation = EnvironmentalObservationDB(
        latitude=observation.latitude,
        longitude=observation.longitude,

        soil_ph=observation.soil_ph,
        soil_organic_carbon=observation.soil_organic_carbon,
        soil_moisture=observation.soil_moisture,

        land_use=observation.land_use,
        land_cover=observation.land_cover,

        species_richness=observation.species_richness,
        habitat_diversity=observation.habitat_diversity,

        temperature=observation.temperature,
        rainfall=observation.rainfall,

        pollution_index=observation.pollution_index,
        deforestation_rate=observation.deforestation_rate
    )

    db.add(db_observation)
    db.commit()
    db.refresh(db_observation)

    return {
        "message": "Environmental observation stored successfully",
        "id": db_observation.id
    }
