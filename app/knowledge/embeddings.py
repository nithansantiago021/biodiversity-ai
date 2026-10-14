import os
from typing import List
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.config import settings


def get_embedding_model():
    """Initializes HuggingFaceEndpointEmbeddings with API token."""
    hf_token = os.getenv("HF_TOKEN", "")
    return HuggingFaceEndpointEmbeddings(
        model=f"https://router.huggingface.co/hf-inference/models/{settings.EMBEDDING_MODEL_NAME}",
        huggingfacehub_api_token=hf_token if hf_token else None,
    )


def generate_embedding(text: str) -> List[float]:
    """Generates an embedding vector for a single string with safe CI fallback."""
    # Instantly return dummy vector if CI / Fake LLM mode is active
    if os.getenv("USE_FAKE_LLM", "").strip().lower() in ("1", "true", "yes"):
        return [0.0] * 384

    try:
        model = get_embedding_model()
        return model.embed_query(text)
    except Exception as e:
        # Fallback if Hugging Face rate limits or rejects token (402/401/429)
        if "402" in str(e) or "Payment Required" in str(e) or "401" in str(e):
            print(
                "HF Inference API credit limit reached / unauthorized. Returning fallback vector."
            )
            return [0.0] * 384
        raise e


def generate_chunk_embeddings(db, chunks) -> list:
    """Generates embeddings for un-embedded DocumentChunks."""
    unembedded = [c for c in chunks if c.embedding is None]
    if not unembedded:
        return chunks

    texts = [c.chunk_text for c in unembedded]

    if os.getenv("USE_FAKE_LLM", "").strip().lower() in ("1", "true", "yes"):
        embeddings = [[0.0] * 384 for _ in texts]
    else:
        try:
            model = get_embedding_model()
            embeddings = model.embed_documents(texts)
        except Exception as e:
            if "402" in str(e) or "Payment Required" in str(e) or "401" in str(e):
                print(
                    "HF Inference API credit limit reached. Returning fallback vectors."
                )
                embeddings = [[0.0] * 384 for _ in texts]
            else:
                raise e

    for chunk, emb in zip(unembedded, embeddings):
        chunk.embedding = emb

    db.commit()
    return chunks
