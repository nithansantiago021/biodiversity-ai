from app.database import SessionLocal
from app.knowledge.ingestion import create_document, create_document_chunks
from app.knowledge.embeddings import generate_chunk_embeddings
from app.knowledge.retrieval import search_similar_chunks


def test_search_similar_chunks_returns_ranked_provenance():
    db = SessionLocal()
    document = None
    chunks = []

    try:
        document = create_document(
            db,
            title="Soil Biology Handbook",
            source="FAO",
            organization="Food and Agriculture Organization",
            document_type="report",
        )

        pages = [
            {
                "page_number": 1,
                "text": "Earthworms and enchytraeids regulate soil structure and aeration.",
            },
            {
                "page_number": 2,
                "text": "Mycorrhizal fungi facilitate plant phosphorus uptake in forest ecosystems.",
            },
        ]

        chunks = create_document_chunks(
            db,
            document,
            pages,
            chunk_size=200,
            chunk_overlap=50,
        )

        # Generate embeddings for test chunks
        generate_chunk_embeddings(db, chunks)

        # Query specifically about mycorrhizal fungi
        query = "Which organisms assist with fungal phosphorus absorption?"
        results = search_similar_chunks(db, query, top_k=1)

        assert len(results) == 1
        top_result = results[0]

        # Verify correct snippet ranking via reranker
        assert "Mycorrhizal fungi" in top_result["chunk_text"]
        assert top_result["page_number"] == 2

        # Verify rich provenance structure
        assert top_result["provenance"]["document_title"] == "Soil Biology Handbook"
        assert (
            top_result["provenance"]["organization"]
            == "Food and Agriculture Organization"
        )
        assert "rerank_score" in top_result
        assert isinstance(top_result["rerank_score"], float)

    finally:
        for chunk in chunks:
            db.delete(chunk)

        db.flush()

        if document:
            db.delete(document)

        db.commit()
        db.close()
