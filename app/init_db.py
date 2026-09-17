from app.database import engine, Base
from app.models.db_models import EnvironmentalObservation, Document, DocumentChunk

print("Creating database tables...")

Base.metadata.create_all(bind=engine)

print("Database tables created successfully.")