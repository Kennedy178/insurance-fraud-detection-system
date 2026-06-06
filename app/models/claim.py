# app/models/claim.py
"""
Pydantic schema for claim input.
Field names and value formats match the original dataset exactly —
feature_service.py depends on these precise names and string values.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ClaimInput(BaseModel):
    """
    Raw claim fields exactly as the original dataset columns.
    feature_service.py reads these keys by name — do not rename them.
    """

    # ── Temporal ──────────────────────────────────────────────
    Month: Optional[str]              = Field(None, description="Month of accident e.g. Jan, Feb")
    WeekOfMonth: Optional[int]        = Field(None, ge=1, le=5)
    DayOfWeek: Optional[str]          = Field(None, description="Monday, Tuesday ...")
    MonthClaimed: Optional[str]       = Field(None, description="Month claim filed e.g. Jan")
    WeekOfMonthClaimed: Optional[int] = Field(None, ge=1, le=5)
    DayOfWeekClaimed: Optional[str]   = Field(None, description="Monday, Tuesday ...")

    # ── Claimant ──────────────────────────────────────────────
    Sex: Optional[str]                = Field(None, description="Male or Female")
    MaritalStatus: Optional[str]      = Field(None, description="Single, Married, Widow, Divorced")
    Age: Optional[int]                = Field(None, ge=16, le=100)
    DriverRating: Optional[int]       = Field(None, ge=1, le=4)

    # ── Vehicle ───────────────────────────────────────────────
    Make: Optional[str]               = Field(None, description="Honda, Toyota, BMW ...")
    VehicleCategory: Optional[str]    = Field(None, description="Sedan, Sport, Utility")
    VehiclePrice: Optional[str]       = Field(None, description="e.g. 20,000 to 29,000")
    AgeOfVehicle: Optional[str]       = Field(None, description="e.g. 3 years, new, more than 7")
    AgeOfPolicyHolder: Optional[str]  = Field(None, description="e.g. 26 to 30, 31 to 35")

    # ── Policy ────────────────────────────────────────────────
    PolicyType: Optional[str]         = Field(None, description="e.g. Sedan - Liability")
    BasePolicy: Optional[str]         = Field(None, description="Liability, Collision, All Perils")
    Deductible: Optional[int]         = Field(None, ge=0, le=10000)

    # ── Incident ──────────────────────────────────────────────
    AccidentArea: Optional[str]       = Field(None, description="Urban or Rural")
    Fault: Optional[str]              = Field(None, description="Policy Holder or Third Party")
    AgentType: Optional[str]          = Field(None, description="Internal or External")

    # ── Claim history ─────────────────────────────────────────
    PoliceReportFiled: Optional[str]  = Field(None, description="Yes or No")
    WitnessPresent: Optional[str]     = Field(None, description="Yes or No")
    PastNumberOfClaims: Optional[str] = Field(None, description="none, 1, 2 to 4, more than 4")
    NumberOfSuppliments: Optional[str]= Field(None, description="none, 1 to 2, 3 to 5, more than 5")
    NumberOfCars: Optional[str]       = Field(None, description="1 vehicle, 2 vehicles, 3 to 4 ...")

    # ── Fields with special characters (kept as-is) ───────────
    # Note: JSON keys can contain colons and hyphens fine
    Days_Policy_Accident: Optional[str] = Field(
        None, alias="Days:Policy-Accident",
        description="none, 1 to 7, 8 to 15, 15 to 30, more than 30"
    )
    Days_Policy_Claim: Optional[str] = Field(
        None, alias="Days:Policy-Claim",
        description="none, 8 to 15, 15 to 30, more than 30"
    )
    AddressChange_Claim: Optional[str] = Field(
        None, alias="AddressChange-Claim",
        description="no change, under 6 months, 1 year, 2 to 3 years, 4 to 8 years"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
    "example": {
        "Month": "Jun",
        "WeekOfMonth": 1,
        "DayOfWeek": "Thursday",
        "MonthClaimed": "Jun",
        "WeekOfMonthClaimed": 1,
        "DayOfWeekClaimed": "Thursday",
        "Sex": "Male",
        "MaritalStatus": "Married",
        "Age": 28,
        "DriverRating": 2,
        "Make": "Honda",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20,000 to 29,000",
        "AgeOfVehicle": "3 years",
        "AgeOfPolicyHolder": "26 to 30",
        "PolicyType": "Sedan - Collision",
        "BasePolicy": "Collision",
        "Deductible": 400,
        "AccidentArea": "Rural",
        "Fault": "Policy Holder",
        "AgentType": "External",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "PastNumberOfClaims": "none",
        "NumberOfSuppliments": "more than 5",
        "NumberOfCars": "1 vehicle",
        "Days:Policy-Accident": "8 to 15",
        "Days:Policy-Claim": "8 to 15",
        "AddressChange-Claim": "1 year"
    }
}
    }