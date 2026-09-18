import uuid
from app.database import SessionLocal
from app.agent.memory import save_chat_message, get_chat_history
from app.models.db_models import ChatMessage


def test_chat_memory_save_and_retrieve():
    db = SessionLocal()
    session_id = f"test-session-{uuid.uuid4()}"

    try:
        # Save multi-turn conversation
        save_chat_message(
            db, session_id, "user", "Biodiversity is declining on my land."
        )
        save_chat_message(
            db,
            session_id,
            "assistant",
            "Can you provide soil organic carbon %, rainfall pattern, and land use type?",
        )
        save_chat_message(
            db,
            session_id,
            "user",
            "Soil carbon is 0.3%, rainfall is 850mm, and land use is agriculture.",
        )

        history = get_chat_history(db, session_id, limit=5)

        assert len(history) == 3
        # Index 0 = Oldest Turn, Index 2 = Latest Turn
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Biodiversity is declining on my land."
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert (
            history[2]["content"]
            == "Soil carbon is 0.3%, rainfall is 850mm, and land use is agriculture."
        )

    finally:
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        db.commit()
        db.close()
