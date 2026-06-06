# app/db/schemas.py
"""
Pydantic schemas for database read/write operations.

These are SEPARATE from the API input schemas in app/models/.
Separation of concerns:
  - app/models/claim.py    → what the API receives from the caller
  - app/models/prediction.py → what the API returns to the caller
  - app/db/schemas.py      → what gets written to / read from PostgreSQL

This prevents coupling between API contract and DB schema,
allowing each to evolve independently.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ── Claim schemas ────

class ClaimCreate(BaseModel):
    """Data needed to create a Claim row in PostgreSQL."""

    # Temporal
    month: str | None                   = None
    week_of_month: int | None           = None
    day_of_week: str | None             = None
    month_claimed: str | None           = None
    week_of_month_claimed: int | None   = None
    day_of_week_claimed: str | None     = None

    # Claimant
    sex: str | None                     = None
    marital_status: str | None          = None
    age: int | None                     = None
    driver_rating: int | None           = None

    # Vehicle
    make: str | None                    = None
    vehicle_category: str | None        = None
    vehicle_price: str | None           = None
    age_of_vehicle: str | None          = None
    age_of_policy_holder: str | None    = None

    # Policy
    policy_type: str | None             = None
    base_policy: str | None             = None
    deductible: int | None              = None

    # Incident
    accident_area: str | None           = None
    fault: str | None                   = None
    agent_type: str | None              = None

    # Claim history
    police_report_filed: str | None     = None
    witness_present: str | None         = None
    past_number_of_claims: str | None   = None
    number_of_suppliments: str | None   = None
    number_of_cars: str | None          = None
    days_policy_accident: str | None    = None
    days_policy_claim: str | None       = None
    address_change_claim: str | None    = None


class ClaimRead(ClaimCreate):
    """Full claim row as returned from DB — includes generated fields."""
    id: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Prediction schemas ───

class PredictionCreate(BaseModel):
    """Data needed to create a Prediction row in PostgreSQL."""
    claim_id: str

    is_fraud: bool
    fraud_probability: float
    risk_score: int
    risk_level: str
    confidence: str
    recommendation: str

    model_name: str
    model_version: str
    deployed_threshold: float
    algorithm: str
    inference_ms: float

    risk_factors: list[dict[str, Any]] | None = None


class PredictionRead(PredictionCreate):
    """Full prediction row as returned from DB."""
    id: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Audit log schemas ───

class AuditLogCreate(BaseModel):
    """Data needed to write an audit log entry."""
    claim_id: str
    action: str
    ip_address: str | None             = None
    user_agent: str | None             = None
    details: dict[str, Any] | None     = None


class AuditLogRead(AuditLogCreate):
    """Full audit log row as returned from DB."""
    id: str
    created_at: datetime

    model_config = {"from_attributes": True}