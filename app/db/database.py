# app/db/database.py
"""
Database engine and session management.

Design decisions:
- async engine + asyncpg driver → non-blocking DB calls, consistent with
  FastAPI's async request handlers
- async_sessionmaker with expire_on_commit=False → prevents lazy-load errors
  after commit (common pitfall)
- get_db() yields a session per request, always closes on exit (even on error)
- pool_pre_ping=True → detects stale connections before use (production safety)
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator

from app.core.config import settings
from loguru import logger


# ── Async engine ────

def _build_async_url(url: str) -> str:
    """
    Ensure the DATABASE_URL uses the asyncpg driver.
    Handles both postgresql:// and postgresql+asyncpg:// inputs.
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = create_async_engine(
    _build_async_url(settings.DATABASE_URL),
    echo=settings.DEBUG,           # logs SQL in development only
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,            # validate connections before use
    pool_recycle=1800,             # recycle connections every 30 min
)


# ── Session factory ──

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # safe for async — avoids lazy-load after commit
    autocommit=False,
    autoflush=False,
)


# ── Declarative base ───

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    All models in models.py inherit from this.
    """
    pass


# ── FastAPI dependency ──

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session per request.

    Usage in endpoint:
        async def my_endpoint(db: AsyncSession = Depends(get_db)):
            ...

    Guarantees:
    - Session is always closed after request, even on exception
    - Rolls back on unhandled exception to prevent dirty state
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()