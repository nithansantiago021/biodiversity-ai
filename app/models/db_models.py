from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.database import Base
from datetime import datetime
from zoneinfo import ZoneInfo


class EnvironmentalObservation(Base):
    __tablename__ = "environmental_observations"

    id = Column(Integer, primary_key=True, index=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    soil_ph = Column(Float, nullable=False)
    soil_organic_carbon = Column(Float, nullable=False)
    soil_moisture = Column(Float, nullable=False)

    land_use = Column(String, nullable=False)
    land_cover = Column(String, nullable=False)

    species_richness = Column(Float, nullable=False)
    habitat_diversity = Column(Float, nullable=False)

    temperature = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)

    pollution_index = Column(Float, nullable=False)
    deforestation_rate = Column(Float, nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String, nullable=False)
    source = Column(String, nullable=False)
    organization = Column(String, nullable=False)

    publication_date = Column(Date, nullable=True)

    url = Column(String, nullable=True)
    document_type = Column(String, nullable=False)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)

    chunk_text = Column(String, nullable=False)
    page_number = Column(Integer, nullable=True)

    chunk_metadata = Column(JSON, nullable=True)

    # Vector embedding column for similarity search
    embedding = Column(Vector(384), nullable=True)

    # Relationship back to parent document
    document = relationship("Document", back_populates="chunks")


Document.chunks = relationship(
    "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now(ZoneInfo("Asia/Kolkata")))
