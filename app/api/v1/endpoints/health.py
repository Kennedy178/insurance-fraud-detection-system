# app/api/v1/endpoints/health.py  (fixed Day 16)
"""
Health and status endpoints.
GET /health          — lightweight liveness check (<10ms target)
GET /api/v1/status   — detailed system status
"""

import time
from fastapi import APIRouter
from loguru import logger
from app.models.prediction import HealthResponse
from app.core.config import settings

router = APIRouter()

_startup_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Lightweight liveness check. Returns 200 if API is running.",
    tags=["Health"],
)
async def health_check():
    from app.services.ml_service import get_ml_service
    try:
        service = get_ml_service()
        model_loaded = service.is_ready()
    except Exception:
        model_loaded = False

    return HealthResponse(
        status         = "ok" if model_loaded else "degraded",
        version        = settings.APP_VERSION,
        model_loaded   = model_loaded,
        environment    = settings.ENVIRONMENT,
        uptime_seconds = round(time.time() - _startup_time, 1),
    )


@router.get(
    "/api/v1/status",
    summary="Detailed system status",
    description="Full system status including model info and config.",
    tags=["Health"],
)
async def system_status():
    from app.services.ml_service import get_ml_service
    try:
        service      = get_ml_service()
        model_loaded = service.is_ready()
        detector     = service.detector
        model_info   = {
            "loaded"   : model_loaded,
            "name"     : detector.metadata.get("model_name"),
            "version"  : detector.metadata.get("model_version"),
            "threshold": detector.deployed_threshold,
            "features" : detector.metadata.get("n_features"),
        }
    except Exception:
        model_loaded = False
        model_info   = {"loaded": False}

    status = {
        "api": {
            "status"     : "ok",
            "version"    : settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime_s"   : round(time.time() - _startup_time, 1),
        },
        "model"   : model_info,
        "database": {"status": "not_configured"},  # Day 18
        "cache"   : {"status": "not_configured"},  # Day 19
    }

    logger.info(f"Status check | model_loaded={model_loaded}")
    return status