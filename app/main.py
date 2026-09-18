from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.db_models import EnvironmentalObservation as EnvironmentalObservationDB
from app.models.schemas import EnvironmentalObservation, ChatRequest, ChatResponse
from app.agent.workflow import biodiversity_agent
from app.config import settings
from app.agent.memory import save_chat_message, get_chat_history


app = FastAPI(
    title="Biodiversity AI",
    description="AI-powered environmental intelligence system",
    version="0.1.0"
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "message": "Biodiversity AI system online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "biodiversity-ai"
    }


@app.post("/observations", status_code=status.HTTP_201_CREATED)
def create_observation(
    observation: EnvironmentalObservation,
    db: Session = Depends(get_db),
):
    db_obs = EnvironmentalObservationDB(**observation.model_dump())
    db.add(db_obs)
    db.commit()
    db.refresh(db_obs)
    return {"id": db_obs.id, "message": "Environmental observation created successfully"}


@app.post("/observations/{observation_id}/recommendations")
def get_recommendation_for_observation(
    observation_id: int,
    db: Session = Depends(get_db),
):
    """
    Triggers the multi-step LangGraph Agent workflow for a stored EnvironmentalObservation ID.
    Executes grounding, two-stage vector retrieval, LLM synthesis, and provenance self-validation.
    """
    obs = (
        db.query(EnvironmentalObservationDB)
        .filter(EnvironmentalObservationDB.id == observation_id)
        .first()
    )
    if not obs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environmental observation with ID {observation_id} not found.",
        )

    initial_state = {
        "observation_id": obs.id,
        "observation": obs,
        "db": db,
        "generated_queries": [],
        "scientific_evidence": [],
        "recommendation_response": {},
        "validation_passed": False,
        "errors": [],
    }

    final_state = biodiversity_agent.invoke(initial_state)

    if not final_state["validation_passed"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Agent provenance validation failed.",
                "errors": final_state["errors"],
            },
        )

    return final_state["recommendation_response"]

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    # 1. Save incoming user message
    save_chat_message(db, payload.session_id, "user", payload.message)

    # 2. If an inline observation payload was sent, persist it to DB so it acquires an .id
    db_obs = None
    if payload.observation:
        db_obs = EnvironmentalObservationDB(**payload.observation.model_dump())
        db.add(db_obs)
        db.commit()
        db.refresh(db_obs)

    # 3. Construct initial Agent state payload with DB instance
    initial_state = {
        "observation_id": db_obs.id if db_obs else None,
        "observation": db_obs,
        "user_query": payload.message,
        "db": db,
        "needs_clarification": False,
        "clarification_question": None,
        "generated_queries": [],
        "scientific_evidence": [],
        "recommendation_response": {},
        "validation_passed": False,
        "errors": [],
    }

    # 4. Invoke LangGraph agent execution graph
    final_state = biodiversity_agent.invoke(initial_state)

    # 5. Handle Clarification vs Recommendation execution branches
    if final_state.get("needs_clarification"):
        reply_content = final_state.get(
            "clarification_question",
            "Can you provide soil organic carbon %, rainfall pattern, soil pH, or land use type?",
        )
        save_chat_message(db, payload.session_id, "assistant", reply_content)
        history = get_chat_history(db, payload.session_id)
        return ChatResponse(
            session_id=payload.session_id,
            response_type="clarification",
            content=reply_content,
            data=None,
            chat_history=history,
        )

    rec_data = final_state.get("recommendation_response", {})
    summary = rec_data.get("ecological_summary", "Grounded recommendation generated successfully.")
    save_chat_message(db, payload.session_id, "assistant", summary)
    history = get_chat_history(db, payload.session_id)

    return ChatResponse(
        session_id=payload.session_id,
        response_type="recommendation",
        content=summary,
        data=rec_data,
        chat_history=history,
    )