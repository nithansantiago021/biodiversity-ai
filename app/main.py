from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
import csv
import io
from sqlalchemy.orm import Session
from typing import List, Optional, Any, Dict

from langchain_core.messages import HumanMessage

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

    # Standalone one-off request -- not part of a conversational thread.
    config = {"configurable": {"thread_id": f"observation-{observation_id}"}}
    final_state = biodiversity_agent.invoke(initial_state, config=config)

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
    # 1. Save incoming user message (durable audit trail in Postgres)
    save_chat_message(db, payload.session_id, "user", payload.message)

    # 2. If an inline observation payload was sent, persist it to DB so it acquires an .id
    db_obs = None
    if payload.observation:
        db_obs = EnvironmentalObservationDB(**payload.observation.model_dump())
        db.add(db_obs)
        db.commit()
        db.refresh(db_obs)

    # 3. Construct the turn's input state. 
    initial_state = {
        "observation_id": db_obs.id if db_obs else None,
        "observation": db_obs,
        "user_query": payload.message,
        "messages": [HumanMessage(content=payload.message)],
        "db": db,
        "needs_clarification": False,
        "clarification_question": None,
        "generated_queries": [],
        "scientific_evidence": [],
        "recommendation_response": {},
        "validation_passed": False,
        "errors": [],
    }

    # 4. Invoke LangGraph agent execution graph, scoped to this session's thread
    config = {"configurable": {"thread_id": payload.session_id}}
    final_state = biodiversity_agent.invoke(initial_state, config=config)

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
    summary = rec_data.get("ecological_summary") or final_state.get("final_response") or "Grounded recommendation generated successfully."
    save_chat_message(db, payload.session_id, "assistant", summary)
    history = get_chat_history(db, payload.session_id)

    return ChatResponse(
        session_id=payload.session_id,
        response_type="recommendation",
        content=summary,
        data=rec_data,
        chat_history=history,
    )

@app.post("/observations/upload-csv", status_code=status.HTTP_201_CREATED)
def upload_csv_observations(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Batch ingests environmental observations from an uploaded CSV file.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a .csv file."
        )

    content = file.file.read().decode("utf-8")
    csv_reader = csv.DictReader(io.StringIO(content))

    created_records = []
    for row in csv_reader:
        obs_data = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "soil_ph": float(row["soil_ph"]),
            "soil_organic_carbon": float(row["soil_organic_carbon"]),
            "soil_moisture": float(row["soil_moisture"]),
            "land_use": str(row["land_use"]),
            "land_cover": str(row["land_cover"]),
            "species_richness": float(row["species_richness"]),
            "habitat_diversity": float(row["habitat_diversity"]),
            "temperature": float(row["temperature"]),
            "rainfall": float(row["rainfall"]),
            "pollution_index": float(row["pollution_index"]),
            "deforestation_rate": float(row["deforestation_rate"]),
        }
        db_obs = EnvironmentalObservationDB(**obs_data)
        db.add(db_obs)
        created_records.append(db_obs)

    db.commit()
    return {
        "message": f"Successfully ingested {len(created_records)} environmental observations.",
        "record_count": len(created_records)
    }