# app/db/__init__.py
"""
Database package — exposes the three things every module needs:
  - Base        : SQLAlchemy declarative base (for models)
  - get_db      : FastAPI dependency for async sessions
  - engine      : async engine (used by Alembic and startup)
"""

from app.db.database import Base, get_db, engine

__all__ = ["Base", "get_db", "engine"]