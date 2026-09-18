from typing import List, Dict
from sqlalchemy.orm import Session
from app.models.db_models import ChatMessage


def save_chat_message(
    db: Session, session_id: str, role: str, content: str
) -> ChatMessage:
    """Persists a single turn of conversation (user or assistant) into PostgreSQL."""
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_chat_history(
    db: Session, session_id: str, limit: int = 10
) -> List[Dict[str, str]]:
    """Retrieves chronological chat history for a given session_id."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    # Reverse list so it returns chronological order (oldest -> newest)
    return [{"role": msg.role, "content": msg.content} for msg in reversed(messages)]
