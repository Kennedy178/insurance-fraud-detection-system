# app/api/v1/router.py  (updated Day 16)
from fastapi import APIRouter
from app.api.v1.endpoints import health
from app.api.v1.endpoints import prediction

api_router = APIRouter()

# Health — root level
api_router.include_router(health.router)

# Predictions — Day 16
api_router.include_router(prediction.router, tags=["Predictions"])

# Day 17: claims CRUD
# from app.api.v1.endpoints import claims
# api_router.include_router(claims.router, prefix="/api/v1/claims", tags=["Claims"])

# Day 19: analytics
# from app.api.v1.endpoints import analytics
# api_router.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])