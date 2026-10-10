from sqlalchemy import text
from app.database import engine, Base
from app.models.db_models import (  # noqa: F401
    EnvironmentalObservation,
    Document,
    DocumentChunk,
    ChatMessage,
)


def init_db():
    print("Enabling pgvector extension...")
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS VECTOR;"))
        connection.commit()

    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()
