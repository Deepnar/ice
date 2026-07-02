"""
Database engine and session factory for the ICE FastAPI proxy.
Uses synchronous SQLAlchemy – all database operations are fast enough
that they don't need async.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from src.api.config import settings

# Create the engine using the DATABASE_URL from config
engine = create_engine(
    settings.database_url,
    pool_size=50,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Session factory – call SessionLocal() to get a new session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.
    The session is automatically closed when the request finishes.

    Usage in a route:
        @app.get("/something")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()