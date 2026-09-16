from sqlalchemy import Column, Integer, String, Float
from app.database import Base


class EnvironmentalObservation(Base):
    __tablename__ = "environmental_observations"

    id = Column(Integer, primary_key=True, index=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    soil_ph = Column(Float, nullable=False)
    soil_organic_carbon = Column(Float, nullable=False)
    soil_moisture = Column(Float, nullable=False)

    land_use = Column(String, nullable=False)
    land_cover = Column(String, nullable=False)

    species_richness = Column(Float, nullable=False)
    habitat_diversity = Column(Float, nullable=False)

    temperature = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)

    pollution_index = Column(Float, nullable=False)
    deforestation_rate = Column(Float, nullable=False)
