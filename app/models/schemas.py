from pydantic import BaseModel, Field

class EnvironmentalObservation(BaseModel):

    #location
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    #soil
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