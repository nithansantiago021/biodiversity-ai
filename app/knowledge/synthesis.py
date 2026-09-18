from typing import Dict, Any
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from app.models.db_models import EnvironmentalObservation
from app.knowledge.grounding import analyze_observation_with_grounding
from app.models.schemas import GroundedRecommendationResponse
from app.llm import get_chat_llm
from app.config import settings

load_dotenv()

SYSTEM_PROMPT = """You are an expert Ecological AI Engine.
Analyze the provided environmental metrics and retrieved scientific evidence.

CRITICAL INSTRUCTIONS:
1. Address the SPECIFIC domain pressures mentioned in the metrics and context (e.g., heavy metals, air pollution, nitrates, soil acidification).
2. DO NOT output generic boilerplate recommendations (like "drip irrigation" or "deforestation monitoring") unless directly relevant to the specific metrics/pressures analyzed.
3. Base all recommendations strictly on the retrieved scientific evidence chunks and cite chunk IDs or document pages for every claim.
4. Summarize key stressors in the ecological_summary field.
"""

HUMAN_PROMPT = """
ENVIRONMENTAL METRICS:
{metrics}

GROUNDED SCIENTIFIC EVIDENCE:
{evidence}

Synthesize these findings into a structured recommendation payload matching the GroundedRecommendationResponse schema.
"""


def generate_grounded_recommendation(
    db: Session,
    observation: EnvironmentalObservation,
) -> Dict[str, Any]:
    """
    Synthesizes grounded scientific evidence into structured recommendations using LangChain + Structured Output LLM.
    Uses Groq when GROQ_API_KEY is set, otherwise falls back to a local
    Ollama model automatically (see app/llm.py::get_chat_llm).
    """
    # 1. Retrieve grounded evidence & provenance from vector store
    grounded_payload = analyze_observation_with_grounding(
        db, observation, top_k_per_query=1
    )

    # 2. Format retrieved evidence blocks for prompt context
    evidence_text_blocks = []
    for idx, ev in enumerate(grounded_payload.get("scientific_evidence", []), start=1):
        prov = ev.get("provenance") or {}
        doc_title = prov.get("document_title") or "Unknown Document"
        page_num = ev.get("page_number") or "N/A"
        chunk_id = ev.get("chunk_id") or "N/A"

        block = (
            f"[{idx}] Chunk ID: {chunk_id} | Doc: {doc_title} | Page: {page_num}\n"
            f"Query Trigger: {ev.get('query_trigger', '')}\n"
            f"Content: {ev.get('chunk_text', '').strip()}\n"
        )
        evidence_text_blocks.append(block)

    evidence_context = (
        "\n".join(evidence_text_blocks)
        if evidence_text_blocks
        else "No specific document chunks retrieved."
    )
    metrics_context = str(grounded_payload.get("metrics", {}))

    # 3. LLM Setup -- Groq if GROQ_API_KEY is present, else local Ollama fallback.
    llm = get_chat_llm(temperature=getattr(settings, "LLM_TEMPERATURE", 0.1))

    # 4. Enforce Pydantic Structured Output
    structured_llm = llm.with_structured_output(GroundedRecommendationResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    chain = prompt | structured_llm

    # 5. Invoke Chain
    result = chain.invoke(
        {
            "metrics": metrics_context,
            "evidence": evidence_context,
        }
    )

    # 6. Normalize output dictionary and preserve observation_id
    if isinstance(result, GroundedRecommendationResponse):
        result.observation_id = observation.id
        output_dict = result.model_dump()
    elif isinstance(result, dict):
        result["observation_id"] = observation.id
        output_dict = result
    else:
        output_dict = {
            "observation_id": observation.id,
            "ecological_summary": str(result),
            "primary_pressures": [],
            "recommendations": [],
        }

    return output_dict
