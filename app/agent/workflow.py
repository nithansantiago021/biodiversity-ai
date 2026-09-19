import re
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage

from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation
from app.knowledge.grounding import generate_observation_queries
from app.knowledge.retrieval import search_similar_chunks
from app.knowledge.synthesis import generate_grounded_recommendation
from app.llm import get_chat_llm


# 1. State Schema Definition
class BiodiversityAgentState(TypedDict, total=False):
    observation_id: Optional[int]
    user_query: Optional[str]
    user_prompt: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    needs_clarification: bool
    clarification_count: int
    clarification_question: Optional[str]
    generated_queries: List[str]
    scientific_evidence: List[Dict[str, Any]]
    recommendation_response: Dict[str, Any]
    final_response: Optional[str]
    validation_passed: bool
    errors: List[str]


# 2. Helper Utilities
def _load_observation(
    db, observation_id: Optional[int]
) -> Optional[EnvironmentalObservation]:
    """Fetch a persisted observation by id using a caller-supplied session."""
    if observation_id is None:
        return None
    return db.get(EnvironmentalObservation, observation_id)


def _latest_user_text(state: BiodiversityAgentState) -> str:
    """Best-effort extraction of the current user text."""
    text = state.get("user_query") or state.get("user_prompt") or ""
    if not text and state.get("messages"):
        for msg in reversed(state.get("messages") or []):
            if getattr(msg, "type", None) == "human":
                content = msg.content
                if isinstance(content, str):
                    text = content
                else:
                    text = str(content)
                break
    return text or ""


def _infer_metric_queries(text: str) -> List[str]:
    text_l = text.lower()
    queries: List[str] = []

    ph_match = re.search(r"ph\s*(?:of|=|is)?\s*(\d+(?:\.\d+)?)", text_l)
    if ph_match:
        queries.append(
            f"Effects of soil pH {ph_match.group(1)} on biodiversity and nutrient availability."
        )

    soc_match = re.search(
        r"(?:soc|organic carbon)\s*(?:of|=|is)?\s*(\d+(?:\.\d+)?)%?", text_l
    )
    if soc_match:
        queries.append(
            f"Impacts of soil organic carbon at {soc_match.group(1)}% on soil health and biodiversity."
        )

    if any(k in text_l for k in ["rain", "monsoon", "drought"]):
        queries.append(
            "Effects of low rainfall and drought stress on soil biodiversity and species survival."
        )

    if "monoculture" in text_l:
        queries.append(
            "Impacts of monoculture cropping on habitat diversity and species richness."
        )

    if "deforest" in text_l:
        queries.append(
            "Consequences of deforestation on habitat fragmentation and species richness."
        )

    if "pollut" in text_l:
        queries.append(
            "Ecotoxicological effects of pollution on soil fauna and biodiversity."
        )

    return queries


# 3. Node Functions
def evaluate_input_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    count = state.get("clarification_count", 0)
    has_observation = state.get("observation_id") is not None
    query_text = _latest_user_text(state).lower()

    bypass_phrases = [
        "dont know",
        "don't know",
        "no metrics",
        "give me all",
        "tell me",
        "what does",
        "effects of",
    ]
    user_requested_direct_search = any(
        phrase in query_text for phrase in bypass_phrases
    )

    if has_observation or count >= 2 or user_requested_direct_search:
        return {"needs_clarification": False, "clarification_question": None}

    essential_keywords = [
        "carbon",
        "ph",
        "rainfall",
        "moisture",
        "temperature",
        "nitrogen",
        "soil",
        "so2",
        "sulfur",
        "sulphur",
        "tds",
        "bis",
    ]
    has_metric_mention = any(kw in query_text for kw in essential_keywords)

    if not has_metric_mention:
        return {"needs_clarification": True, "clarification_question": None}

    return {"needs_clarification": False, "clarification_question": None}


def ask_clarification_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    user_query = _latest_user_text(state)
    current_count = state.get("clarification_count", 0)

    llm = get_chat_llm()

    clarify_prompt = f"""You are an Ecological AI Engine.
The user submitted the following prompt:
"{user_query}"

This request lacks essential site context or specific environmental parameters needed to perform precise scientific retrieval.

INSTRUCTIONS:
1. Ask 2 brief, highly relevant clarification questions specific to their query domain.
2. Direct the user to provide specific numerical parameters (e.g., pH, organic carbon %, soil texture, or contaminant type).
"""

    response = llm.invoke(clarify_prompt)
    content = response.content
    if isinstance(content, str):
        question = content.strip()
    else:
        question = str(content).strip()

    return {
        "clarification_count": current_count + 1,
        "clarification_question": question,
        "final_response": question,
        # Return ONLY the new message -- the add_messages reducer appends it
        # to the persisted history. Returning old + new here would duplicate
        # the whole history every turn.
        "messages": [AIMessage(content=question)],
    }


def ground_metrics_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """
    Node 1: Builds the vector-search query set.
    """
    observation_id = state.get("observation_id")
    user_text = _latest_user_text(state)

    if observation_id is not None:
        db = SessionLocal()
        try:
            obs = _load_observation(db, observation_id)
        finally:
            db.close()
        queries = generate_observation_queries(obs) if obs is not None else []
        combined_topic_query = user_text
    else:
        queries = _infer_metric_queries(user_text)
        combined_topic_query = (
            user_text
            or "sulfur dioxide SO2 atmospheric pollution effects on lake water quality"
        )

    queries.insert(0, combined_topic_query)
    unique_queries = list(dict.fromkeys(q for q in queries if q))
    return {"generated_queries": unique_queries}


def retrieve_evidence_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node 2: Executes two-stage retrieval (pgvector + Cross-Encoder) for queries."""
    db = SessionLocal()
    queries = state.get("generated_queries", [])
    evidence_items = []
    seen_chunk_ids = set()

    try:
        for q in queries:
            retrieved_chunks = search_similar_chunks(db, query_text=q, top_k=2)
            for item in retrieved_chunks:
                chunk_id = item.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    evidence_items.append(
                        {
                            "query_trigger": q,
                            "chunk_id": chunk_id,
                            "page_number": item.get("page_number"),
                            "rerank_score": item.get("rerank_score"),
                            "chunk_text": item.get("chunk_text"),
                            "provenance": item.get("provenance"),
                        }
                    )
    finally:
        db.close()

    return {"scientific_evidence": evidence_items}


def synthesize_recommendation_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node 3: Synthesizes structured recommendations or text RAG response."""
    observation_id = state.get("observation_id")
    evidence = state.get("scientific_evidence", [])
    user_query = _latest_user_text(state)

    if observation_id is not None:
        db = SessionLocal()
        try:
            obs = _load_observation(db, observation_id)
            if obs is None:
                raise ValueError(
                    f"No EnvironmentalObservation found for id {observation_id}."
                )
            recommendation_data = generate_grounded_recommendation(db, obs)
        finally:
            db.close()

        if isinstance(recommendation_data, dict):
            recommendation_data["observation_id"] = obs.id
            if (
                "ecological_summary" not in recommendation_data
                and "summary" in recommendation_data
            ):
                recommendation_data["ecological_summary"] = recommendation_data[
                    "summary"
                ]

        summary = (
            recommendation_data.get("ecological_summary")
            or recommendation_data.get("summary")
            or str(recommendation_data)
        )

        return {
            "observation_id": obs.id,
            "recommendation_response": recommendation_data,
            "final_response": summary,
            "messages": [AIMessage(content=summary)],
        }

    # Branch B: Pure Text/CLI RAG Workflow (no EnvironmentalObservation attached)
    if not evidence:
        refusal = (
            "Notice: There is insufficient evidence or matching records available in the ingested database "
            "for this specific query and parameter set. Please refine your query or consult additional scientific literature."
        )
        return {
            "recommendation_response": {},
            "final_response": refusal,
            "messages": [AIMessage(content=refusal)],
        }

    llm = get_chat_llm()

    context_str = "\n\n".join(
        [
            f"[Chunk {item.get('chunk_id')} | Doc: {item.get('provenance', {}).get('document_title')} | Page: {item.get('page_number')}]\n{item.get('chunk_text')}"
            for item in evidence
        ]
    )

    synthesis_prompt = f"""You are an Ecological AI Engine.
User Query: "{user_query}"

Scientific Context retrieved from database:
{context_str}

CRITICAL INSTRUCTIONS:
1. Answer the query directly using ONLY the provided scientific context.
2. Formulate 2-3 structured, actionable recommendations. For EACH one, state explicitly:
   - What to do
   - Why it works (scientific reasoning)
   - Which environmental metric(s) it improves
   - Time horizon (short-term / medium-term / long-term)
   - A confidence level (low / medium / high)
3. Cite document titles, page numbers, and chunk IDs explicitly for every claim.
"""

    response = llm.invoke(synthesis_prompt)
    answer = response.content

    return {
        "recommendation_response": {"summary": answer, "evidence": evidence},
        "final_response": answer,
        "messages": [AIMessage(content=answer)],
    }


def validate_provenance_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node 4: Validates chunk provenance citations against retrieved evidence."""
    evidence_chunk_ids = {
        item["chunk_id"]
        for item in state.get("scientific_evidence", [])
        if "chunk_id" in item
    }
    rec_response = state.get("recommendation_response", {})

    errors = []
    for rec in rec_response.get("recommendations", []):
        for citation in rec.get("citations", []):
            cited_id = citation.get("chunk_id")
            if cited_id and cited_id not in evidence_chunk_ids:
                errors.append(
                    f"Invalid citation: chunk_id {cited_id} missing from evidence."
                )

    return {"validation_passed": len(errors) == 0, "errors": errors}


# 4. Routing Logic
def route_input(state: BiodiversityAgentState) -> str:
    if state.get("needs_clarification"):
        return "ask_clarification"
    return "ground_metrics"


# 5. Graph Assembly
def build_biodiversity_agent():
    workflow = StateGraph(BiodiversityAgentState)

    workflow.add_node("evaluate_input", evaluate_input_node)
    workflow.add_node("ask_clarification", ask_clarification_node)
    workflow.add_node("ground_metrics", ground_metrics_node)
    workflow.add_node("retrieve_evidence", retrieve_evidence_node)
    workflow.add_node("synthesize_recommendation", synthesize_recommendation_node)
    workflow.add_node("validate_provenance", validate_provenance_node)

    workflow.set_entry_point("evaluate_input")

    workflow.add_conditional_edges(
        "evaluate_input",
        route_input,
        {
            "ask_clarification": "ask_clarification",
            "ground_metrics": "ground_metrics",
        },
    )

    workflow.add_edge("ask_clarification", END)
    workflow.add_edge("ground_metrics", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "synthesize_recommendation")
    workflow.add_edge("synthesize_recommendation", "validate_provenance")
    workflow.add_edge("validate_provenance", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


biodiversity_agent = build_biodiversity_agent()
