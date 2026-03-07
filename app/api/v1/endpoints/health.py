# app/api/v1/endpoints/health.py
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

# Startup time — set when the app starts
_startup_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Lightweight liveness check. Returns 200 if API is running.",
    tags=["Health"],
)
async def health_check():
    """
    Primary health endpoint.
    Load balancers and monitoring tools ping this — must be <10ms.
    Does NOT check DB or model — use /api/v1/status for that.
    """
    from app.main import fraud_detector  # import here to avoid circular

    model_loaded = fraud_detector is not None and fraud_detector._loaded

    logger.debug("Health check called")

    return HealthResponse(
        status        = "ok" if model_loaded else "degraded",
        version       = settings.APP_VERSION,
        model_loaded  = model_loaded,
        environment   = settings.ENVIRONMENT,
        uptime_seconds = round(time.time() - _startup_time, 1),
    )


@router.get(
    "/api/v1/status",
    summary="Detailed system status",
    description="Full system status including model info and config.",
    tags=["Health"],
)
async def system_status():
    """
    Detailed status — model version, threshold, environment config.
    Use this for monitoring dashboards, not for load balancer pings.
    """
    from app.main import fraud_detector

    model_loaded = fraud_detector is not None and fraud_detector._loaded

    status = {
        "api": {
            "status"     : "ok",
            "version"    : settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime_s"   : round(time.time() - _startup_time, 1),
        },
        "model": {
            "loaded"   : model_loaded,
            "name"     : fraud_detector.metadata.get("model_name") if model_loaded else None,
            "version"  : fraud_detector.metadata.get("model_version") if model_loaded else None,
            "threshold": fraud_detector.deployed_threshold if model_loaded else None,
            "features" : fraud_detector.metadata.get("n_features") if model_loaded else None,
        },
        "database": {
            "status": "not_configured",  # updated Day 18
        },
        "cache": {
            "status": "not_configured",  # updated Day 19
        },
    }

    logger.info(f"Status check | model_loaded={model_loaded}")
    return status