import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from app.config import settings


def get_chat_llm(temperature: Optional[float] = None):
    """
    Single place that decides which chat LLM backs the agent.

    - If GROQ_API_KEY is set in the environment, use Groq (settings.LLM_MODEL_NAME).
    - Otherwise, fall back to a local Ollama model (settings.OLLAMA_MODEL_NAME,
      default "qwen3:8b") so the agent still works offline / without an API key.

    Every LLM call site in the agent (clarification questions, free-text
    synthesis, structured recommendation synthesis) should go through this
    function instead of instantiating ChatGroq/ChatOllama directly, so the
    fallback behavior is consistent everywhere.
    """
    temp = settings.LLM_TEMPERATURE if temperature is None else temperature
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return ChatGroq(
            model_name=settings.LLM_MODEL_NAME,
            temperature=temp,
            api_key=api_key,
        )

    # No Groq key available -- use local Ollama instead of crashing.
    return ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        temperature=temp,
        base_url=settings.OLLAMA_BASE_URL,
    )
