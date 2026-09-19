import os
import re
from typing import Optional, Type

from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from app.config import settings


class _FakeAIMessage:
    """Minimal stand-in for an AIMessage -- callers only ever read `.content`."""

    def __init__(self, content: str):
        self.content = content


_CHUNK_REF_PATTERN = re.compile(
    r"Chunk ID:\s*(?P<chunk_id>\d+)\s*\|\s*Doc:\s*(?P<doc>[^|]+?)\s*\|\s*Page:\s*(?P<page>\S+)"
)


def _build_fake_structured_response(schema: Type, prompt_value):
    """
    Builds a schema-valid fake GroundedRecommendationResponse for CI.

    Pulls real chunk_id/document_title/page_number references out of the
    rendered prompt text -- synthesis.py embeds them as
    "Chunk ID: X | Doc: Y | Page: Z" for every evidence block -- so the fake
    citation points at evidence that genuinely exists, instead of an invented
    id that would fail app.agent.workflow.validate_provenance_node's
    citation check.
    """
    text = (
        prompt_value.to_string()
        if hasattr(prompt_value, "to_string")
        else str(prompt_value)
    )
    match = _CHUNK_REF_PATTERN.search(text)

    if match:
        citation = {
            "document_title": match.group("doc").strip(),
            "page_number": (
                int(match.group("page")) if match.group("page").isdigit() else 1
            ),
            "chunk_id": int(match.group("chunk_id")),
        }
    else:
        # No evidence was retrieved for this call -- cite nothing real rather
        # than fabricating an id that would fail provenance validation.
        citation = {
            "document_title": "No evidence retrieved",
            "page_number": 1,
            "chunk_id": 0,
        }

    payload = {
        "observation_id": 0,  # overwritten by the caller with the real observation id
        "ecological_summary": "Fake CI summary: key stressors identified from the provided metrics and evidence.",
        "primary_pressures": ["low soil organic carbon", "soil acidification"],
        "recommendations": [
            {
                "title": "Introduce legume-based cover crops",
                "description": (
                    "Fake CI recommendation standing in for a real LLM call: "
                    "cover cropping raises soil organic carbon and supports "
                    "microbial diversity."
                ),
                "impacted_metrics": ["soil_organic_carbon", "species_richness"],
                "time_horizon": "medium-term",
                "citations": [citation],
            }
        ],
    }
    return schema(**payload)


class FakeChatLLM:
    """
    Stand-in for ChatGroq/ChatOllama used when USE_FAKE_LLM is set, so tests
    (and CI) never hit a real LLM API or require a local Ollama server.
    Implements only the two call shapes the agent actually uses.
    """

    def __init__(self, temperature: Optional[float] = None):
        self.temperature = temperature

    def invoke(self, prompt) -> _FakeAIMessage:
        return _FakeAIMessage(
            "Fake CI response. Could you share the soil pH, Soil Organic "
            "Carbon %, and rainfall pattern for your site?"
        )

    def with_structured_output(self, schema: Type):
        return RunnableLambda(
            lambda prompt_value: _build_fake_structured_response(schema, prompt_value)
        )


def get_chat_llm(temperature: Optional[float] = None):
    """
    Single place that decides which chat LLM backs the agent.

    - If USE_FAKE_LLM is set (e.g. "1"), always use FakeChatLLM. This is what
      CI sets so GitHub Actions never calls a real API and needs no secrets.
    - Else, if GROQ_API_KEY is set, use Groq (settings.LLM_MODEL_NAME).
    - Otherwise, fall back to a local Ollama model (settings.OLLAMA_MODEL_NAME,
      default "qwen3:8b") so the agent still works offline / without an API key.

    Every LLM call site in the agent (clarification questions, free-text
    synthesis, structured recommendation synthesis) should go through this
    function instead of instantiating a chat model directly, so this
    switching logic is consistent everywhere.
    """
    temp = settings.LLM_TEMPERATURE if temperature is None else temperature

    if os.getenv("USE_FAKE_LLM", "").strip().lower() in ("1", "true", "yes"):
        return FakeChatLLM(temperature=temp)

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
