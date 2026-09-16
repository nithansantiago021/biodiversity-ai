from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = (
    "postgresql+psycopg2://"
    "biodiversity_user:"
    "biodiversity_password@"
    "localhost:5433/"
    "biodiversity"
)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False,
    bind=engine
)

Base = declarative_base()

