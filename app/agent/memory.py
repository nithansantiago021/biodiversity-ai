from typing import List, Dict, cast
from sqlalchemy import desc, select
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
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(desc(ChatMessage.id))
        .limit(limit)
    )
    messages = db.scalars(stmt).all()

    return [
        {
            "role": cast(str, msg.role),
            "content": cast(str, msg.content),
        }
        for msg in reversed(messages)
    ]
