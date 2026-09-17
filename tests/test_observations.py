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

def test_get_recommendation_endpoint_success(client):
    # 1. Post a valid observation
    payload = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "soil_ph": 5.2,
        "soil_organic_carbon": 0.4,
        "soil_moisture": 12.0,
        "land_use": "agriculture",
        "land_cover": "cropland",
        "species_richness": 14.0,
        "habitat_diversity": 0.3,
        "temperature": 28.5,
        "rainfall": 850.0,
        "pollution_index": 0.2,
        "deforestation_rate": 0.08,
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 201
    obs_id = response.json()["id"]

    # 2. Trigger recommendation synthesis endpoint
    rec_response = client.post(f"/observations/{obs_id}/recommendations")
    assert rec_response.status_code == 200
    data = rec_response.json()

    assert data["observation_id"] == obs_id
    assert "ecological_summary" in data
    assert len(data["recommendations"]) > 0
    assert "citations" in data["recommendations"][0]
