# app/main.py  (updated Day 16)
"""
FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs     ← Swagger UI
    http://localhost:8000/redoc    ← ReDoc
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1.router import api_router
from app.services.ml_service import init_ml_service

# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):

    # ── STARTUP ───────────────────────────────────────────────
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    try:
        service = init_ml_service()
        logger.info(
            f"ML service ready | "
            f"model={service.detector.metadata['model_name']} | "
            f"version={service.detector.metadata['model_version']} | "
            f"threshold={service.detector.deployed_threshold:.4f}"
        )
    except Exception as e:
        logger.error(f"ML service failed to start: {e}")
        logger.warning("API starting in degraded mode — predictions will fail")

    logger.info("Startup complete — ready to serve requests")
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────
    logger.info("Shutting down")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = settings.APP_NAME,
    version     = settings.APP_VERSION,
    description = (
        "Production-ready insurance fraud detection API. "
        "Powered by XGBoost with business-optimal threshold calibration. "
        "Catches 72.5% of fraudulent claims — $294,000 verified saving "
        "per 2,300 claims vs default threshold."
    ),
    docs_url  = "/docs",
    redoc_url = "/redoc",
    lifespan  = lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins_list,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers     = ["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    if settings.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc} | path={request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "path": str(request.url.path)},
    )

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(api_router)

# ── Root ───────────────────────────────────────────────────────────────────
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