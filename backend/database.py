from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL

Base = declarative_base()


@lru_cache(maxsize=1)
def get_engine():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set in the environment. "
            "Point it at your Postgres/Supabase connection string."
        )
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db():
    """FastAPI dependency that yields a DB session and closes it afterward."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they don't exist yet. Called once on app startup."""
    from . import models  # noqa: F401  (registers models on Base before create_all)

    Base.metadata.create_all(bind=get_engine())