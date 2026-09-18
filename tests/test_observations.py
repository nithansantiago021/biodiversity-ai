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

def test_upload_csv_observations_success(client):
    csv_content = (
        "latitude,longitude,soil_ph,soil_organic_carbon,soil_moisture,land_use,land_cover,species_richness,habitat_diversity,temperature,rainfall,pollution_index,deforestation_rate\n"
        "13.0827,80.2707,5.5,0.45,15.0,agriculture,cropland,12.0,0.35,29.0,900.0,0.2,0.05\n"
        "12.9716,77.5946,6.2,0.80,20.0,forest,dense_forest,45.0,0.85,24.5,1200.0,0.1,0.01\n"
    )

    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    response = client.post("/observations/upload-csv", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["record_count"] == 2
    assert "Successfully ingested" in data["message"]