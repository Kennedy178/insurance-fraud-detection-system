# app/main.py
"""
FastAPI application entry point.

Startup sequence:
  1. Create DB tables if they don't exist (idempotent)
  2. Load ML model + scaler + feature names into memory
  3. Start accepting requests

Shutdown sequence:
  1. Dispose DB connection pool cleanly
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.database import engine, Base

# Import all models so SQLAlchemy registers them before create_all
import app.db.models  # noqa: F401


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager that runs startup and shutdown logic.
    Replaces deprecated on_event("startup") / on_event("shutdown").
    """
    # ── STARTUP ───────────────────────────────────────────────────
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Create all DB tables (idempotent — safe to run on every startup)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("DB | Tables verified / created successfully")
    except Exception as e:
        logger.error(f"DB | Failed to create tables: {e}")
        logger.warning("DB | Continuing startup — check DATABASE_URL in .env")

    # Load ML model
    # Load ML model
    try:
        from app.services.ml_service import init_ml_service, get_ml_service
        init_ml_service()
        service = get_ml_service()
        if service.is_ready():
            logger.info(f"ML | Model loaded: {settings.MODEL_NAME} v{settings.MODEL_VERSION}")
        else:
            logger.warning("ML | Model not loaded — predictions will fail")
    except Exception as e:
        logger.error(f"ML | Model load error: {e}")

    logger.info("Startup complete — accepting requests")

    yield  # ← application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────
    logger.info("Shutting down...")
    await engine.dispose()
    logger.info("DB | Connection pool disposed")
    logger.info("Shutdown complete")


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = settings.APP_NAME,
    version     = settings.APP_VERSION,
    description = (
        "Production-grade insurance fraud detection API. "
        "XGBoost model trained on 15,420 claims. "
        "ROC-AUC 0.781 | 72.5% fraud detection rate | $294,000 verified savings."
    ),
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)


# ── CORS ───────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins_list,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Security headers middleware ────────────────────────────────────────────────

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    return response


# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code}"
    )
    return response


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc} | path={request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Check server logs."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

from app.api.v1.endpoints.prediction import router as prediction_router
from app.api.v1.endpoints.health     import router as health_router

app.include_router(health_router)
app.include_router(prediction_router)


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name"       : settings.APP_NAME,
        "version"    : settings.APP_VERSION,
        "status"     : "running",
        "docs"       : "/docs",
        "health"     : "/health",
        "environment": settings.ENVIRONMENT,
    }