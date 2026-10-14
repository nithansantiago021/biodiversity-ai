from app.database import SessionLocal
from app.models.db_models import Document
from app.knowledge.ingestion import create_document, create_document_chunks
from app.knowledge.embeddings import generate_chunk_embeddings
from app.knowledge.retrieval import search_similar_chunks


def test_search_similar_chunks_returns_ranked_provenance():
    db = SessionLocal()
    document = None
    chunks = []

    try:
        # Delete leftover test records from previous runs to prevent duplicates
        db.query(Document).filter_by(title="Soil Biology Handbook Test Doc").delete()
        db.commit()

        document = create_document(
            db,
            title="Soil Biology Handbook Test Doc",
            source="FAO_Test",
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

        generate_chunk_embeddings(db, chunks)

        # Retrieve with candidate_pool_size=10000 to scan past 8k DB chunks during mock vector distance ties
        query = "Which organisms assist with fungal phosphorus absorption?"
        results = search_similar_chunks(db, query, top_k=50, candidate_pool_size=10000)

        # Filter results to the created test document
        doc_results = [r for r in results if r["document_id"] == document.id]

        assert len(doc_results) > 0
        top_result = doc_results[0]

        # Verify correct snippet ranking via reranker
        assert "Mycorrhizal fungi" in top_result["chunk_text"]
        assert top_result["page_number"] == 2

        # Verify rich provenance structure
        assert (
            top_result["provenance"]["document_title"]
            == "Soil Biology Handbook Test Doc"
        )
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
