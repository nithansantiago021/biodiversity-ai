from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Config
    APP_NAME: str = "Biodiversity AI Backend"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    
    # LLM & Model Parameters (used when GROQ_API_KEY is available)
    LLM_MODEL_NAME: str = "openai/gpt-oss-120b"
    LLM_TEMPERATURE: float = 0.1

    # Local fallback LLM (used automatically when GROQ_API_KEY is NOT set --
    # see app/llm.py::get_chat_llm). Requires a local Ollama server running
    # with the model pulled: `ollama pull qwen3:8b`.
    OLLAMA_MODEL_NAME: str = "qwen3:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Embedding & Retrieval Parameters
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_TOP_K: int = 2

    # Enable reading from local .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Single global settings instance
settings = Settings()