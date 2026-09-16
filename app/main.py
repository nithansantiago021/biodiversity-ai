from fastapi import FastAPI
from app.models.environment import EnvironmentalObservation

app = FastAPI(
    title = "Biodiversity AI",
    description="AI powered environmental intelligence system",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "Biodiversity AI system online"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "biodiversity-ai"
    }

@app.post("/environment")
def create_environment(
    observation: EnvironmentalObservation
):
    return {
        "message": "Environmental observation received",
        "data": observation
    }