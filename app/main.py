# app/main.py
"""
FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000          → root
    http://localhost:8000/health   → health check
    http://localhost:8000/docs     → Swagger UI  ← most useful during dev
    http://localhost:8000/redoc    → ReDoc
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# ── Make ml_pipeline importable from app/ ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1.router import api_router

# ── Global model instance ──────────────────────────────────────────────────
# Loaded once at startup, reused for all requests — never re-loaded per request
fraud_detector = None


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model into memory.
    Shutdown: clean up resources.
    """
    global fraud_detector

    # ── STARTUP ───────────────────────────────────────────────
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    try:
        from ml_pipeline.inference import FraudDetector
        fraud_detector = FraudDetector(model_dir=settings.MODEL_DIR)
        logger.info(
            f"Model loaded | name={fraud_detector.metadata['model_name']} "
            f"| version={fraud_detector.metadata['model_version']} "
            f"| threshold={fraud_detector.deployed_threshold:.4f}"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model — prediction endpoints will fail")

    logger.info("Startup complete — ready to serve requests")

    yield  # ← app runs here

    # ── SHUTDOWN ──────────────────────────────────────────────
    logger.info("Shutting down — cleaning up resources")
    fraud_detector = None


# ── App instance ───────────────────────────────────────────────────────────
app = FastAPI(
    title       = settings.APP_NAME,
    version     = settings.APP_VERSION,
    description = (
        "Production-ready insurance fraud detection API. "
        "Powered by XGBoost with business-optimal threshold calibration. "
        "Catches 72.5% of fraudulent claims with $294,000 verified savings "
        "per 2,300 claims vs default threshold. "
        "See /docs for interactive API documentation."
    ),
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────────────

# CORS — allow React dev servers and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins_list,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers     = ["*"],
)


# Request logging middleware — logs every request with timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({elapsed:.1f}ms)"
    )
    return response


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    if settings.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


# ── Global exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc} | path={request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error"  : "Internal server error",
            "detail" : "An unexpected error occurred. Check server logs.",
            "path"   : str(request.url.path),
        }
    )


# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(api_router)


# ── Root endpoint ──────────────────────────────────────────────────────────
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