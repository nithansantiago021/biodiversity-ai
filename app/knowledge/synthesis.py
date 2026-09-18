import os
from typing import List
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.models.db_models import EnvironmentalObservation
from app.knowledge.grounding import analyze_observation_with_grounding
from app.models.schemas import (
    GroundedRecommendationResponse,
    ActionableRecommendation,
    CitationSchema,
)
from app.config import settings

load_dotenv()

SYSTEM_PROMPT = """You are an expert Environmental Intelligence AI.
Your task is to analyze structured environmental observation metrics alongside grounded scientific evidence chunks retrieved from PostgreSQL/pgvector.

CRITICAL INSTRUCTIONS:
1. Every recommended intervention MUST strictly cite at least one grounded evidence chunk provided in the context.
2. Ensure citations match the exact document_title, page_number, and chunk_id provided.
3. Keep recommendations actionable, scientifically sound, and clear.
"""

HUMAN_PROMPT = """
ENVIRONMENTAL METRICS:
{metrics}

GROUNDED SCIENTIFIC EVIDENCE:
{evidence}

Synthesize these findings into a structured recommendation payload.
"""


def generate_grounded_recommendation(
    db: Session,
    observation: EnvironmentalObservation,
) -> dict:
    """
    Synthesizes grounded scientific evidence into structured recommendations using LangChain + Groq LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")

    # 1. Retrieve grounded evidence & provenance
    grounded_payload = analyze_observation_with_grounding(
        db, observation, top_k_per_query=1
    )

    # 2. Format evidence for prompt context
    evidence_text_blocks = []
    for idx, ev in enumerate(grounded_payload["scientific_evidence"], start=1):
        block = (
            f"[{idx}] Chunk ID: {ev['chunk_id']} | Doc: {ev['provenance']['document_title']} | Page: {ev['page_number']}\n"
            f"Query Trigger: {ev['query_trigger']}\n"
            f"Content: {ev['chunk_text']}\n"
        )
        evidence_text_blocks.append(block)

    evidence_context = "\n".join(evidence_text_blocks)
    metrics_context = str(grounded_payload["metrics"])

    # 3. LangChain + Groq Structured Output Workflow
    llm = ChatGroq(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        api_key=api_key,
    )

    structured_llm = llm.with_structured_output(GroundedRecommendationResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    chain = prompt | structured_llm

    result: GroundedRecommendationResponse = chain.invoke(
        {
            "metrics": metrics_context,
            "evidence": evidence_context,
        }
    )

    result.observation_id = observation.id
    return result.model_dump()
