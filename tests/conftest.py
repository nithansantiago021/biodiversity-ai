import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """
    Pytest fixture providing a TestClient instance for testing FastAPI endpoints.
    """
    return TestClient(app)
