# app/api/v1/endpoints/prediction.py
"""
Prediction endpoints — Day 16
================================
POST /api/v1/predict          → single claim prediction
POST /api/v1/predict/batch    → batch prediction
GET  /api/v1/model/info       → model metadata
GET  /api/v1/model/features   → feature list
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.models.claim import ClaimInput
from app.models.prediction import (
    PredictionResponse, BatchPredictionResponse, ModelInfoResponse
)
from app.services.ml_service import FraudDetectionService, get_ml_service

router = APIRouter()


def _claim_to_raw_dict(claim: ClaimInput) -> dict:
    """
    Convert ClaimInput Pydantic model → raw dict with original column names
    matching the training dataset exactly.

    API uses snake_case field names; feature_engineering.py used PascalCase
    and original dataset column names. This mapping bridges the two.
    """
    raw = {}

    # Direct mappings — API field → dataset column name
    field_map = {
        'age'                    : 'Age',
        'deductible'             : 'Deductible',
        'driver_rating'          : 'DriverRating',
        'week_of_month'          : 'WeekOfMonth',
        'week_of_month_claimed'  : 'WeekOfMonthClaimed',
        'fault'                  : 'Fault',
        'agent_type'             : 'AgentType',
        'accident_area'          : 'AccidentArea',
        'police_report_filed'    : 'PoliceReportFiled',
        'witnesses'              : 'WitnessPresent',   # mapped below
        'vehicle_category'       : 'VehicleCategory',
        'base_policy'            : 'BasePolicy',
        'policy_type'            : 'PolicyType',
        'make'                   : 'Make',
        'day_of_week'            : 'DayOfWeek',
        'day_of_week_claimed'    : 'DayOfWeekClaimed',
        'claim_date_month'       : 'MonthClaimed',     # month claim filed
        'month_claimed'          : 'MonthClaimed',
        'past_number_of_claims'  : 'PastNumberOfClaims',
        'days_policy_accident'   : 'Days:Policy-Accident',
        'days_policy_claim'      : 'Days:Policy-Claim',
        'number_of_supplements'  : 'NumberOfSuppliments',
        'address_change_claim'   : 'AddressChange-Claim',
        'number_of_cars'         : 'NumberOfCars',
        'year'                   : 'Year',
    }

    claim_dict = claim.model_dump(mode='json', exclude_none=True)

    for api_field, dataset_col in field_map.items():
        if api_field in claim_dict and claim_dict[api_field] is not None:
            raw[dataset_col] = claim_dict[api_field]

    # Special case: witnesses (int) → WitnessPresent (Yes/No string)
    if 'witnesses' in claim_dict:
        raw['WitnessPresent'] = 'Yes' if claim_dict['witnesses'] > 0 else 'No'

    return raw


# ── Single prediction ──────────────────────────────────────────────────────

@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a single insurance claim",
    description=(
        "Submit claim details and receive an instant fraud risk assessment. "
        "Returns fraud probability, risk score (0-100), risk level, "
        "recommendation, and top 5 contributing risk factors."
    ),
    tags=["Predictions"],
)
async def predict_single(
    claim  : ClaimInput,
    service: FraudDetectionService = Depends(get_ml_service),
):
    """
    Primary prediction endpoint — used by the investigator dashboard.

    - Input: ClaimInput JSON body
    - Output: Full PredictionResponse with risk factors and recommendation
    - Target: <100ms end-to-end
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Check /health endpoint.",
        )

    try:
        raw_claim = _claim_to_raw_dict(claim)
        result    = service.predict_claim(raw_claim)
        logger.info(
            f"POST /api/v1/predict | "
            f"risk={result['risk_level']} | "
            f"fraud={result['is_fraud']} | "
            f"{result['inference_ms']}ms"
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs.",
        )


# ── Batch prediction ───────────────────────────────────────────────────────

@router.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse multiple claims in one request",
    description="Submit up to 100 claims for batch fraud analysis.",
    tags=["Predictions"],
)
async def predict_batch(
    claims : list[ClaimInput],
    service: FraudDetectionService = Depends(get_ml_service),
):
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded.",
        )

    if len(claims) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit is 100 claims per request.",
        )

    if len(claims) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch must contain at least 1 claim.",
        )

    try:
        raw_claims = [_claim_to_raw_dict(c) for c in claims]
        result     = service.predict_batch(raw_claims)
        logger.info(
            f"POST /api/v1/predict/batch | "
            f"total={result['total_claims']} | "
            f"fraud={result['fraud_count']} | "
            f"{result['total_ms']}ms"
        )
        return result

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        )


# ── Model info ─────────────────────────────────────────────────────────────

@router.get(
    "/api/v1/model/info",
    summary="Get model metadata and performance metrics",
    tags=["Model"],
)
async def model_info(service: FraudDetectionService = Depends(get_ml_service)):
    """Returns model version, training date, test set metrics, threshold in use."""
    return service.get_model_info()


@router.get(
    "/api/v1/model/features",
    summary="Get list of features used by the model",
    tags=["Model"],
)
async def model_features(service: FraudDetectionService = Depends(get_ml_service)):
    """Returns all 76 feature names and their metadata."""
    return service.get_feature_list()