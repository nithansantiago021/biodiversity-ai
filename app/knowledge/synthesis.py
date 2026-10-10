from typing import Dict, Any, cast
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from app.models.db_models import EnvironmentalObservation
from app.knowledge.grounding import analyze_observation_with_grounding
from app.knowledge.rules import (
    evaluate_rules,
    build_constraints_block,
    apply_post_filters,
)
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
5. For every recommendation, state its confidence (high, medium, low, or insufficient) and its tradeoffs
   (the honest cost, failure mode, or precondition) -- never omit these.
6. Any MANDATORY CONSTRAINTS section below is non-negotiable: it comes from a deterministic rule
   check, not a suggestion. Follow it exactly, even if it changes which intervention leads.
"""

HUMAN_PROMPT = """
ENVIRONMENTAL METRICS:
{metrics}

{constraints}

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

    A deterministic interaction-rules engine (app.knowledge.rules) runs BEFORE this call decides
    anything: it evaluates the observation's variables in plain Python and produces mandatory
    constraints. Those constraints are (1) injected into the LLM's prompt, and (2) enforced again
    on the LLM's own output afterward, so the rule's effect holds even if the LLM ignores the
    prompt. The LLM's job is to phrase the recommendation -- not to decide whether a pollinator
    habitat intervention is safe next to a fixed pesticide schedule.

    Uses Groq when GROQ_API_KEY is set, otherwise falls back to a local Ollama model automatically
    (see app/llm.py::get_chat_llm), or the fake LLM in CI (USE_FAKE_LLM=1).
    """
    # 1. Deterministic rule evaluation -- no LLM involved.
    rule_results = evaluate_rules(observation)
    constraints_block = build_constraints_block(rule_results)

    # 2. Retrieve grounded evidence & provenance from vector store
    grounded_payload = analyze_observation_with_grounding(
        db, observation, top_k_per_query=1
    )

    # 3. Format retrieved evidence blocks for prompt context
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

    # 4. LLM Setup -- Groq if GROQ_API_KEY is present, else local Ollama fallback (or fake in CI).
    llm = get_chat_llm(temperature=getattr(settings, "LLM_TEMPERATURE", 0.1))

    # 5. Enforce Pydantic Structured Output
    structured_llm = llm.with_structured_output(GroundedRecommendationResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    chain = prompt | structured_llm

    # 6. Invoke Chain
    result = chain.invoke(
        {
            "metrics": metrics_context,
            "constraints": constraints_block,
            "evidence": evidence_context,
        }
    )

    # 7. Normalize output dictionary and preserve observation_id
    if isinstance(result, GroundedRecommendationResponse):
        result.observation_id = cast(int, observation.id)
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

    # 8. Deterministic post-filter -- fires regardless of LLM compliance, and
    # records which rules fired for an auditable trace in the response.
    output_dict["recommendations"] = apply_post_filters(
        rule_results, output_dict.get("recommendations", [])
    )
    output_dict["rule_trace"] = [f"{r.rule_id}: {r.reason}" for r in rule_results]

    return output_dict
