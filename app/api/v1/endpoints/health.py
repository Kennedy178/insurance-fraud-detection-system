# app/api/v1/endpoints/health.py
"""
Health and status endpoints.
GET /health          — kept for Render uptime monitoring
GET /api/v1/ping     — frontend uses this (bypasses ad-blocker blocklists)
GET /api/v1/status   — detailed system status
"""

import time
from fastapi import APIRouter
from loguru import logger
from app.models.prediction import HealthResponse
from app.core.config import settings

router = APIRouter()

_startup_time = time.time()


async def _get_health_response() -> HealthResponse:
    """Shared logic for both health endpoints."""
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
    "/health",
    response_model=HealthResponse,
    summary="Health check (for Render monitoring)",
    tags=["Health"],
)
async def health_check():
    return await _get_health_response()


@router.get(
    "/api/v1/ping",
    response_model=HealthResponse,
    summary="Ping (frontend status check — ad-blocker safe)",
    tags=["Health"],
)
async def ping():
    return await _get_health_response()


@router.get(
    "/api/v1/status",
    summary="Detailed system status",
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

    return {
        "api": {
            "status"     : "ok",
            "version"    : settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime_s"   : round(time.time() - _startup_time, 1),
        },
        "model"   : model_info,
        "database": {"status": "configured"},
        "cache"   : {"status": "not_configured"},
    }