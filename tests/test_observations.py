import pytest
from pydantic import ValidationError

from app.models.schemas import EnvironmentalObservation


def test_valid_environmental_observation():

    observation = EnvironmentalObservation(
        latitude=13.0827,
        longitude=80.2707,
        soil_ph=6.8,
        soil_organic_carbon=0.3,
        soil_moisture=15.0,
        land_use="agriculture",
        land_cover="cropland",
        species_richness=12,
        habitat_diversity=0.42,
        temperature=29.0,
        rainfall=720.0,
        pollution_index=0.2,
        deforestation_rate=0.01
    )

    assert observation.soil_ph == 6.8


def test_invalid_soil_ph():

    with pytest.raises(ValidationError):

        EnvironmentalObservation(
            latitude=13.0827,
            longitude=80.2707,
            soil_ph=25,
            soil_organic_carbon=0.3,
            soil_moisture=15.0,
            land_use="agriculture",
            land_cover="cropland",
            species_richness=12,
            habitat_diversity=0.42,
            temperature=29.0,
            rainfall=720.0,
            pollution_index=0.2,
            deforestation_rate=0.01
        )
