from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(autouse=True)
def mock_embedding_calls():
    dummy_vector = [0.0] * 384

    def mock_generate_chunk_embeddings(db, chunks):
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = dummy_vector
        db.commit()
        return chunks

    def mock_score(self, text_pairs):
        scores = []
        for query, text in text_pairs:
            if "Mycorrhizal" in text:
                scores.append(0.99)
            elif "Earthworms" in text:
                scores.append(0.85)
            else:
                scores.append(0.10)
        return scores

    with patch(
        "app.knowledge.embeddings.generate_embedding",
        return_value=dummy_vector,
    ), patch(
        "app.knowledge.embeddings.generate_chunk_embeddings",
        side_effect=mock_generate_chunk_embeddings,
    ), patch(
        "app.knowledge.retrieval.HuggingFaceEndpointCrossEncoder.score",
        autospec=True,
        side_effect=mock_score,
    ):
        yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def knowledge_base():
    """Fast session fixture that bypasses PDF parsing during unit tests."""
    yield
