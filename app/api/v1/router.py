# app/api/v1/router.py
"""
V1 API router — aggregates all endpoint routers.
main.py includes this single router with prefix /api/v1.
Add new endpoint modules here as they are built (Days 16-19).
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health

api_router = APIRouter()

# Health — no prefix, /health lives at root level
api_router.include_router(health.router)

# Day 16: prediction endpoints
# from app.api.v1.endpoints import prediction
# api_router.include_router(prediction.router, prefix="/api/v1", tags=["Predictions"])

# Day 17: claims CRUD
# from app.api.v1.endpoints import claims
# api_router.include_router(claims.router, prefix="/api/v1/claims", tags=["Claims"])

# Day 19: analytics
# from app.api.v1.endpoints import analytics
# api_router.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])