# app/db/crud.py
"""
CRUD operations for the fraud detection system.

All functions are:
- async (non-blocking)
- fully typed
- self-contained (receive a session, do one job, return result)
- safe: never raise unhandled exceptions — let the endpoint handle them

Naming convention:
  create_*   → INSERT and return the created object
  get_*      → SELECT one row (returns None if not found)
  list_*     → SELECT multiple rows with optional filters
  log_audit  → INSERT to audit_log (append-only)
"""

from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.db.models import Claim, Prediction, AuditLog
from app.db.schemas import ClaimCreate, PredictionCreate, AuditLogCreate


# ── Claims ──────

async def create_claim(
    db: AsyncSession,
    claim_data: ClaimCreate,
) -> Claim:
    """
    Insert a new claim row and return the created object.
    Called from the prediction endpoint before scoring.
    """
    claim = Claim(**claim_data.model_dump(exclude_none=False))
    db.add(claim)
    await db.flush()   # flush to get the generated id without full commit
    await db.refresh(claim)
    logger.debug(f"DB | claim created | id={claim.id}")
    return claim


async def get_claim(db: AsyncSession, claim_id: str) -> Claim | None:
    """Fetch a single claim by its UUID. Returns None if not found."""
    result = await db.execute(
        select(Claim).where(Claim.id == claim_id)
    )
    return result.scalar_one_or_none()


# ── Predictions ─────

async def create_prediction(
    db: AsyncSession,
    prediction_data: PredictionCreate,
) -> Prediction:
    """
    Insert a prediction result linked to a claim.
    risk_factors list is stored as JSONB.
    """
    data = prediction_data.model_dump()

    # Convert list of risk factor dicts → store as-is (JSONB handles it)
    prediction = Prediction(**data)
    db.add(prediction)
    await db.flush()
    await db.refresh(prediction)
    logger.debug(
        f"DB | prediction saved | id={prediction.id} "
        f"| fraud={prediction.is_fraud} "
        f"| prob={prediction.fraud_probability:.4f}"
    )
    return prediction


async def get_prediction(
    db: AsyncSession,
    prediction_id: str,
) -> Prediction | None:
    """Fetch a single prediction by its UUID."""
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    return result.scalar_one_or_none()


async def list_recent_predictions(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0,
    fraud_only: bool = False,
) -> list[Prediction]:
    """
    Return recent predictions, newest first.
    Used by the analytics dashboard and claims history endpoints.
    """
    query = select(Prediction).order_by(desc(Prediction.created_at))

    if fraud_only:
        query = query.where(Prediction.is_fraud == True)  # noqa: E712

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_predictions_for_claim(
    db: AsyncSession,
    claim_id: str,
) -> list[Prediction]:
    """Return all predictions for a specific claim (supports re-scoring)."""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.claim_id == claim_id)
        .order_by(desc(Prediction.created_at))
    )
    return list(result.scalars().all())


# ── Analytics aggregates ─────

async def get_prediction_stats(
    db: AsyncSession,
    days: int = 30,
) -> dict[str, Any]:
    """
    Compute summary statistics for the analytics dashboard.
    Covers the last `days` days.

    Returns:
        total_predictions, fraud_count, legitimate_count,
        fraud_rate, avg_probability, high_risk_count
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Total count
    total_result = await db.execute(
        select(func.count(Prediction.id))
        .where(Prediction.created_at >= since)
    )
    total = total_result.scalar_one() or 0

    # Fraud count
    fraud_result = await db.execute(
        select(func.count(Prediction.id))
        .where(Prediction.created_at >= since)
        .where(Prediction.is_fraud == True)  # noqa: E712
    )
    fraud_count = fraud_result.scalar_one() or 0

    # Average fraud probability
    avg_result = await db.execute(
        select(func.avg(Prediction.fraud_probability))
        .where(Prediction.created_at >= since)
    )
    avg_prob = avg_result.scalar_one() or 0.0

    # High risk count (risk_level = HIGH)
    high_result = await db.execute(
        select(func.count(Prediction.id))
        .where(Prediction.created_at >= since)
        .where(Prediction.risk_level == "HIGH")
    )
    high_risk = high_result.scalar_one() or 0

    return {
        "period_days"        : days,
        "total_predictions"  : total,
        "fraud_count"        : fraud_count,
        "legitimate_count"   : total - fraud_count,
        "fraud_rate"         : round(fraud_count / total, 4) if total > 0 else 0.0,
        "avg_fraud_probability": round(float(avg_prob), 4),
        "high_risk_count"    : high_risk,
    }


async def get_daily_fraud_trend(
    db: AsyncSession,
    days: int = 30,
) -> list[dict[str, Any]]:
    """
    Return daily fraud counts for the trend chart.
    Each item: { date: "2026-03-15", total: 12, fraud: 3 }
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Daily total predictions
    total_by_day = await db.execute(
        select(
            func.date(Prediction.created_at).label("day"),
            func.count(Prediction.id).label("total"),
        )
        .where(Prediction.created_at >= since)
        .group_by(func.date(Prediction.created_at))
        .order_by(func.date(Prediction.created_at))
    )
    total_rows = {str(row.day): row.total for row in total_by_day}

    # Daily fraud predictions
    fraud_by_day = await db.execute(
        select(
            func.date(Prediction.created_at).label("day"),
            func.count(Prediction.id).label("fraud"),
        )
        .where(Prediction.created_at >= since)
        .where(Prediction.is_fraud == True)  # noqa: E712
        .group_by(func.date(Prediction.created_at))
        .order_by(func.date(Prediction.created_at))
    )
    fraud_rows = {str(row.day): row.fraud for row in fraud_by_day}

    # Merge
    trend = []
    for day, total in sorted(total_rows.items()):
        trend.append({
            "date"  : day,
            "total" : total,
            "fraud" : fraud_rows.get(day, 0),
        })

    return trend


# ── Audit log ──

async def log_audit(
    db: AsyncSession,
    audit_data: AuditLogCreate,
) -> AuditLog:
    """
    Append an audit log entry. Never updated or deleted after creation.
    Called after every prediction for compliance.
    """
    entry = AuditLog(**audit_data.model_dump(exclude_none=False))
    db.add(entry)
    await db.flush()
    logger.debug(
        f"DB | audit log | claim={audit_data.claim_id} "
        f"| action={audit_data.action}"
    )
    return entry