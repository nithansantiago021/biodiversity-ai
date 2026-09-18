from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class EnvironmentalObservation(BaseModel):

    # location
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    # soil
    soil_ph: float = Field(..., ge=0, le=14)
    soil_organic_carbon: float = Field(..., ge=0)
    soil_moisture: float = Field(..., ge=0)

    # Land
    land_use: str
    land_cover: str

    # Biodiversity
    species_richness: float = Field(..., ge=0)
    habitat_diversity: float = Field(..., ge=0)

    # Climate
    temperature: float
    rainfall: float = Field(..., ge=0)

    # Human Impact
    pollution_index: float = Field(..., ge=0)
    deforestation_rate: float = Field(..., ge=0)


class CitationSchema(BaseModel):
    document_title: str
    page_number: int
    chunk_id: int


class ActionableRecommendation(BaseModel):
    title: str = Field(..., description="Short title of the intervention")
    description: str = Field(..., description="Detailed scientific justification")
    impacted_metrics: List[str] = Field(
        ..., description="Metrics addressed, e.g., ['soil_ph', 'soil_organic_carbon']"
    )
    time_horizon: str = Field(
        ...,
        description="Implementation timeframe: short-term, medium-term, or long-term",
    )
    citations: List[CitationSchema] = Field(
        ..., description="Exact references to grounded evidence chunks used"
    )


class GroundedRecommendationResponse(BaseModel):
    observation_id: int
    ecological_summary: str = Field(
        ..., description="Synthesis of current multi-variable pressures"
    )
    primary_pressures: List[str] = Field(
        ..., description="Key ecological stress drivers identified"
    )
    recommendations: List[ActionableRecommendation]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    observation: Optional[EnvironmentalObservation] = None


class ChatResponse(BaseModel):
    session_id: str
    response_type: str
    content: str
    data: Optional[Dict[str, Any]] = None
    chat_history: List[Dict[str, str]]
