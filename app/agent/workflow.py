from typing import TypedDict, List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, END

from app.models.db_models import EnvironmentalObservation
from app.knowledge.grounding import generate_observation_queries
from app.knowledge.retrieval import search_similar_chunks
from app.knowledge.synthesis import generate_grounded_recommendation


# 1. State Schema Definition
class BiodiversityAgentState(TypedDict):
    observation_id: Optional[int]
    observation: Optional[EnvironmentalObservation]
    user_query: Optional[str]
    db: Session
    needs_clarification: bool
    clarification_question: Optional[str]
    generated_queries: List[str]
    scientific_evidence: List[Dict[str, Any]]
    recommendation_response: Dict[str, Any]
    validation_passed: bool
    errors: List[str]


# 2. Node Functions
def evaluate_input_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """
    Node 0: Gatekeeper node. Checks if structured observation or text query has essential metrics.
    If missing key metrics, routes to clarification.
    """
    obs = state.get("observation")
    query = (state.get("user_query") or "").lower()

    # If structured observation with at least one critical threshold is present
    if obs and (
        obs.soil_organic_carbon is not None
        or obs.soil_ph is not None
        or obs.rainfall is not None
    ):
        return {"needs_clarification": False}

    # If pure text query, check for metric indicators
    essential_keywords = [
        "carbon",
        "ph",
        "rainfall",
        "moisture",
        "temperature",
        "nitrogen",
        "soil",
    ]
    has_metric_mention = any(kw in query for kw in essential_keywords)

    if not has_metric_mention and not obs:
        return {
            "needs_clarification": True,
            "clarification_question": (
                "Can you provide soil organic carbon %, rainfall pattern, soil pH, or land use type "
                "so I can generate a scientifically grounded recommendation?"
            ),
        }

    return {"needs_clarification": False}


def ask_clarification_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node A: Returns the clarifying question payload when inputs are vague."""
    return {}


def ground_metrics_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node 1: Evaluates environmental metrics and generates targeted search queries."""
    obs = state["observation"]
    queries = generate_observation_queries(obs)
    return {"generated_queries": queries}


def retrieve_evidence_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node 2: Executes two-stage vector retrieval (pgvector + Cross-Encoder) for queries."""
    db = state["db"]
    queries = state["generated_queries"]
    evidence_items = []

    for query in queries:
        retrieved_chunks = search_similar_chunks(db, query_text=query, top_k=1)
        for item in retrieved_chunks:
            evidence_items.append(
                {
                    "query_trigger": query,
                    "chunk_id": item["chunk_id"],
                    "page_number": item["page_number"],
                    "rerank_score": item["rerank_score"],
                    "chunk_text": item["chunk_text"],
                    "provenance": item["provenance"],
                }
            )

    return {"scientific_evidence": evidence_items}


def synthesize_recommendation_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node 3: Synthesizes structured recommendations via LLM synthesis engine."""
    db = state["db"]
    obs = state["observation"]
    recommendation_data = generate_grounded_recommendation(db, obs)
    return {"recommendation_response": recommendation_data}


def validate_provenance_node(state: BiodiversityAgentState) -> Dict[str, Any]:
    """Node 4: Agent self-validation checking that all cited chunk_ids match retrieved evidence."""
    evidence_chunk_ids = {item["chunk_id"] for item in state["scientific_evidence"]}
    rec_response = state["recommendation_response"]

    errors = []
    for rec in rec_response.get("recommendations", []):
        for citation in rec.get("citations", []):
            cited_id = citation.get("chunk_id")
            if cited_id not in evidence_chunk_ids:
                errors.append(
                    f"Invalid citation: chunk_id {cited_id} not present in retrieved evidence."
                )

    validation_passed = len(errors) == 0
    return {"validation_passed": validation_passed, "errors": errors}


# 3. Routing Logic
def route_input(state: BiodiversityAgentState) -> str:
    if state.get("needs_clarification"):
        return "ask_clarification"
    return "ground_metrics"


# 4. Graph Assembly
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

    return workflow.compile()


biodiversity_agent = build_biodiversity_agent()
