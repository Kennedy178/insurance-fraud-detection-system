# app/db/models.py
"""
SQLAlchemy ORM models for the fraud detection system.

Tables:
  - claims       : one row per analysed insurance claim (raw inputs)
  - predictions  : one row per ML prediction result (linked to claim)
  - audit_log    : append-only compliance trail (every prediction action)

Design decisions:
- UUID primary keys → globally unique, safe for distributed systems / Render
- JSONB for risk_factors → flexible, queryable, no schema changes needed
  when risk factor format evolves
- Separate claims + predictions → allows re-scoring a claim without losing
  original prediction history
- audit_log is INSERT-only (never UPDATE/DELETE) → compliance requirement
- All timestamps use timezone-aware UTC via func.now()
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


# ── Helper: server-side UTC timestamp ──

def _now() -> datetime:
    """Python-side UTC default (used as fallback; DB uses server_default)."""
    from datetime import timezone
    return datetime.now(timezone.utc)


# ── Claims table ────

class Claim(Base):
    """
    Stores the raw claim inputs submitted to the prediction endpoint.
    One row per unique claim submission.
    """
    __tablename__ = "claims"

    # Primary key — UUID generated in Python for predictability
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Temporal fields ───
    month: Mapped[str | None]               = mapped_column(String(20))
    week_of_month: Mapped[int | None]       = mapped_column(Integer)
    day_of_week: Mapped[str | None]         = mapped_column(String(20))
    month_claimed: Mapped[str | None]       = mapped_column(String(20))
    week_of_month_claimed: Mapped[int | None] = mapped_column(Integer)
    day_of_week_claimed: Mapped[str | None] = mapped_column(String(20))

    # ── Claimant fields ───
    sex: Mapped[str | None]                 = mapped_column(String(10))
    marital_status: Mapped[str | None]      = mapped_column(String(20))
    age: Mapped[int | None]                 = mapped_column(Integer)
    driver_rating: Mapped[int | None]       = mapped_column(Integer)

    # ── Vehicle fields ───
    make: Mapped[str | None]                = mapped_column(String(50))
    vehicle_category: Mapped[str | None]    = mapped_column(String(30))
    vehicle_price: Mapped[str | None]       = mapped_column(String(50))
    age_of_vehicle: Mapped[str | None]      = mapped_column(String(30))
    age_of_policy_holder: Mapped[str | None] = mapped_column(String(30))

    # ── Policy fields ────
    policy_type: Mapped[str | None]         = mapped_column(String(50))
    base_policy: Mapped[str | None]         = mapped_column(String(30))
    deductible: Mapped[int | None]          = mapped_column(Integer)

    # ── Incident fields ───
    accident_area: Mapped[str | None]       = mapped_column(String(20))
    fault: Mapped[str | None]               = mapped_column(String(30))
    agent_type: Mapped[str | None]          = mapped_column(String(20))

    # ── Claim history fields ───
    police_report_filed: Mapped[str | None]     = mapped_column(String(5))
    witness_present: Mapped[str | None]         = mapped_column(String(5))
    past_number_of_claims: Mapped[str | None]   = mapped_column(String(20))
    number_of_suppliments: Mapped[str | None]   = mapped_column(String(20))
    number_of_cars: Mapped[str | None]          = mapped_column(String(20))
    days_policy_accident: Mapped[str | None]    = mapped_column(String(20))
    days_policy_claim: Mapped[str | None]       = mapped_column(String(20))
    address_change_claim: Mapped[str | None]    = mapped_column(String(20))

    # ── Metadata ─────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationship ────
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="claim",
        cascade="all, delete-orphan",
        lazy="select",
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="claim",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # ── Indexes ───
    __table_args__ = (
        Index("ix_claims_created_at", "created_at"),
        Index("ix_claims_agent_type", "agent_type"),
        Index("ix_claims_fault", "fault"),
    )

    def __repr__(self) -> str:
        return f"<Claim id={self.id} age={self.age} fault={self.fault}>"


# ── Predictions table ────

class Prediction(Base):
    """
    Stores the ML model's output for each claim.
    Linked to Claim via foreign key.
    One claim can have multiple predictions (re-scoring over time).
    """
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Foreign key to claims ─────
    claim_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Prediction outputs ────
    is_fraud: Mapped[bool]                  = mapped_column(Boolean, nullable=False)
    fraud_probability: Mapped[float]        = mapped_column(Float, nullable=False)
    risk_score: Mapped[int]                 = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str]                 = mapped_column(String(10), nullable=False)
    confidence: Mapped[str]                 = mapped_column(String(10), nullable=False)
    recommendation: Mapped[str]             = mapped_column(Text, nullable=False)

    # ── Model metadata ─────
    model_name: Mapped[str]                 = mapped_column(String(50), nullable=False)
    model_version: Mapped[str]              = mapped_column(String(20), nullable=False)
    deployed_threshold: Mapped[float]       = mapped_column(Float, nullable=False)
    algorithm: Mapped[str]                  = mapped_column(String(50), nullable=False)
    inference_ms: Mapped[float]             = mapped_column(Float, nullable=False)

    # ── Top risk factors (JSONB for flexibility) ───
    risk_factors: Mapped[dict | None]       = mapped_column(JSONB)

    # ── Timestamp ─────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationship ───
    claim: Mapped["Claim"] = relationship("Claim", back_populates="predictions")

    # ── Indexes ────
    __table_args__ = (
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_fraud_probability", "fraud_probability"),
        Index("ix_predictions_risk_level", "risk_level"),
        Index("ix_predictions_is_fraud", "is_fraud"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} "
            f"fraud={self.is_fraud} "
            f"prob={self.fraud_probability:.4f}>"
        )


# ── Audit log table ───

class AuditLog(Base):
    """
    Append-only compliance trail.
    Every prediction action is recorded here — never updated or deleted.
    Required for insurance regulatory compliance.
    """
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    claim_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Action fields ──
    action: Mapped[str]         = mapped_column(String(50), nullable=False)
    # e.g. "prediction_created", "batch_prediction", "manual_review"

    ip_address: Mapped[str | None]  = mapped_column(String(45))
    user_agent: Mapped[str | None]  = mapped_column(String(200))

    # ── Flexible payload ──
    details: Mapped[dict | None]    = mapped_column(JSONB)
    # stores fraud_probability, risk_level, model_version at time of action

    # ── Timestamp ───
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationship ──
    claim: Mapped["Claim"] = relationship("Claim", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_log_created_at", "created_at"),
        Index("ix_audit_log_action", "action"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action}>"