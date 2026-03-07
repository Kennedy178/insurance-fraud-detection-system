# app/core/config.py
"""
Central configuration — all settings loaded from .env via Pydantic BaseSettings.
Every other module imports `settings` from here. Never read os.environ directly.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

# Project root = two levels up from this file (app/core/config.py)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):

    # ── Application ───────────────────────────────────────────
    APP_NAME: str = "Insurance Fraud Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development | production

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── CORS — include both Vite (5173) and CRA (3000) ────────
    ALLOWED_ORIGINS: str = (
        "http://localhost:5173,"
        "http://localhost:3000,"
        "http://localhost:8080"
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_detection"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 0

    # ── Redis ─────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # ── ML Model ──────────────────────────────────────────────
    MODEL_DIR: str = str(ROOT_DIR / "models")
    MODEL_NAME: str = "fraud_detector_v1"
    MODEL_VERSION: str = "1.0.0"

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = str(ROOT_DIR / "logs")
    LOG_ROTATION: str = "500 MB"
    LOG_RETENTION: str = "30 days"

    # ── Rate limiting (Phase 3 future) ────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance — loaded once at startup.
    Use: from app.core.config import get_settings; settings = get_settings()
    """
    return Settings()


# Module-level singleton for convenience
settings = get_settings()