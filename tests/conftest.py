import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.knowledge.ingestion import run_ingestion_pipeline


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def knowledge_base():
    db = SessionLocal()
    try:
        run_ingestion_pipeline(db, raw_data_dir="data/raw_data")
    finally:
        db.close()
    yield
