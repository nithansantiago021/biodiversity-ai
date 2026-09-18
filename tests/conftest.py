import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import engine, Base

# Import the database models explicitly from db_models (NO wildcard imports)
import app.models.db_models as db_models


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Automatically create all tables in the database before running tests."""
    # Base now has full knowledge of environmental_observations, documents,
    # document_chunks, and chat_messages
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    """
    Pytest fixture providing a TestClient instance for testing FastAPI endpoints.
    """
    return TestClient(app)
