import uuid
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_chat_endpoint_triggers_clarification_on_vague_input():
    session_id = f"test-chat-{uuid.uuid4()}"
    payload = {
        "session_id": session_id,
        "message": "Biodiversity is declining on my land.",
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == session_id
    assert data["response_type"] == "clarification"
    assert (
        "Soil Organic Carbon" in data["content"]
        or "pH" in data["content"]
        or "rainfall" in data["content"]
    )
    assert len(data["chat_history"]) == 2


def test_chat_endpoint_processes_full_observation(knowledge_base):
    session_id = f"test-chat-{uuid.uuid4()}"
    payload = {
        "session_id": session_id,
        "message": "Here are my site metrics for analysis.",
        "observation": {
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
        },
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == session_id
    assert data["response_type"] == "recommendation"
    assert data["data"] is not None
    assert len(data["chat_history"]) == 2