# app/models/claim.py
"""
Pydantic schemas for claim input and output.

ClaimInput covers the core raw fields an investigator would submit.
The feature engineering pipeline (app/services/feature_service.py, Day 16)
transforms these into the 76 features the model expects.

Field names match the original dataset columns before encoding —
the API accepts human-readable values, not one-hot encoded arrays.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ── Enums for constrained string fields ───────────────────────────────────

class FaultEnum(str, Enum):
    policy_holder = "Policy Holder"
    third_party   = "Third Party"

class PolicyTypeEnum(str, Enum):
    sedan_liability  = "Sedan - Liability"
    sedan_collision  = "Sedan - Collision"
    sport_liability  = "Sport - Liability"
    sport_collision  = "Sport - Collision"
    utility_liability = "Utility - Liability"
    utility_collision = "Utility - Collision"

class VehicleCategoryEnum(str, Enum):
    sedan   = "Sedan"
    sport   = "Sport"
    utility = "Utility"

class BasePolicyEnum(str, Enum):
    liability  = "Liability"
    collision  = "Collision"
    all_perils = "All Perils"

class AccidentAreaEnum(str, Enum):
    urban = "Urban"
    rural = "Rural"

class IncidentCauseEnum(str, Enum):
    rear_collision    = "Rear Collision"
    left_turn         = "Left Turn"
    right_turn        = "Right Turn"
    front_collision   = "Front Collision"
    parked_bike       = "Parked Car"
    other             = "Other"

class AgentTypeEnum(str, Enum):
    internal = "Internal"
    external = "External"

class AddressChangeEnum(str, Enum):
    no_change   = "no change"
    under_6_months = "under 6 months"
    one_year    = "1 year"
    two_to_three = "2 to 3 years"
    four_to_eight = "4 to 8 years"

class NumberOfCarsEnum(str, Enum):
    one    = "1 vehicle"
    two    = "2 vehicles"
    three  = "3 to 4"
    five   = "5 to 8"
    more   = "more than 8"


# ── Main input schema ──────────────────────────────────────────────────────

class ClaimInput(BaseModel):
    """
    Raw claim fields submitted by the investigator via API.
    All fields are optional with sensible defaults — the model
    handles missing features gracefully (filled with 0 after encoding).
    Required fields are marked explicitly.
    """

    # ── Policy & Vehicle ──────────────────────────────────────
    policy_number: Optional[str] = Field(
        None, description="Policy number for tracking (not used in model)"
    )
    vehicle_category: Optional[VehicleCategoryEnum] = Field(
        None, description="Type of vehicle"
    )
    base_policy: Optional[BasePolicyEnum] = Field(
        None, description="Base policy type"
    )
    policy_type: Optional[PolicyTypeEnum] = Field(
        None, description="Full policy type (vehicle + coverage)"
    )
    deductible: Optional[int] = Field(
        None, ge=0, le=10000, description="Policy deductible amount ($)"
    )

    # ── Claimant ──────────────────────────────────────────────
    age: Optional[int] = Field(
        None, ge=16, le=100, description="Claimant age"
    )
    driver_rating: Optional[int] = Field(
        None, ge=1, le=4, description="Driver risk rating (1=best, 4=worst)"
    )

    # ── Incident ──────────────────────────────────────────────
    accident_area: Optional[AccidentAreaEnum] = Field(
        None, description="Urban or Rural"
    )
    fault: Optional[FaultEnum] = Field(
        None, description="Who is at fault"
    )
    incident_cause: Optional[IncidentCauseEnum] = Field(
        None, description="Cause of incident"
    )
    number_of_cars: Optional[NumberOfCarsEnum] = Field(
        None, description="Number of cars involved"
    )
    number_of_supplements: Optional[int] = Field(
        None, ge=0, le=20, description="Number of supplement claims"
    )
    address_change_claim: Optional[AddressChangeEnum] = Field(
        None, description="Address changes in claim period"
    )

    # ── Claim details ─────────────────────────────────────────
    claim_date_month: Optional[int] = Field(
        None, ge=1, le=12, description="Month claim was filed (1-12)"
    )
    week_of_month_claimed: Optional[int] = Field(
        None, ge=1, le=5, description="Week of month claim was filed"
    )
    day_of_week_claimed: Optional[str] = Field(
        None, description="Day of week claim was filed"
    )
    month_claimed: Optional[int] = Field(
        None, ge=1, le=12, description="Month of incident (1-12)"
    )
    week_of_month: Optional[int] = Field(
        None, ge=1, le=5, description="Week of month of incident"
    )
    day_of_week: Optional[str] = Field(
        None, description="Day of week of incident"
    )

    # ── Agent ─────────────────────────────────────────────────
    agent_type: Optional[AgentTypeEnum] = Field(
        None, description="Internal or external agent handling the claim"
    )

    # ── Additional features ───────────────────────────────────
    past_number_of_claims: Optional[str] = Field(
        None, description="Previous claims history"
    )
    police_report_filed: Optional[str] = Field(
        None, description="Was a police report filed? (Yes/No)"
    )
    witnesses: Optional[int] = Field(
        None, ge=0, le=10, description="Number of witnesses"
    )
    vehicle_price: Optional[str] = Field(
        None, description="Vehicle price range"
    )
    days_policy_accident: Optional[str] = Field(
        None, description="Days between policy start and accident"
    )
    days_policy_claim: Optional[str] = Field(
        None, description="Days between policy start and claim"
    )
    year: Optional[int] = Field(
        None, ge=1990, le=2030, description="Year of claim"
    )
    make: Optional[str] = Field(
        None, description="Vehicle make"
    )

    @field_validator('age')
    @classmethod
    def age_must_be_valid(cls, v):
        if v is not None and (v < 16 or v > 100):
            raise ValueError('Age must be between 16 and 100')
        return v

    @field_validator('deductible')
    @classmethod
    def deductible_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Deductible must be non-negative')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "policy_number": "POL-2024-001",
                "vehicle_category": "Sedan",
                "base_policy": "Liability",
                "policy_type": "Sedan - Liability",
                "deductible": 400,
                "age": 34,
                "driver_rating": 3,
                "accident_area": "Urban",
                "fault": "Policy Holder",
                "incident_cause": "Rear Collision",
                "agent_type": "External",
                "claim_date_month": 3,
                "witnesses": 1,
                "police_report_filed": "No",
            }
        }
    }