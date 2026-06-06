# app/api/v1/endpoints/prediction.py
"""
Prediction endpoints — Day 16 + Day 18 (DB integration)
=========================================================
POST /api/v1/predict          → single claim prediction (saved to DB)
POST /api/v1/predict/batch    → batch prediction (each saved to DB)
GET  /api/v1/model/info       → model metadata
GET  /api/v1/model/features   → feature list
GET  /api/v1/predictions      → recent predictions from DB
GET  /api/v1/stats            → analytics summary from DB
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.claim import ClaimInput
from app.models.prediction import (
    PredictionResponse,
    BatchPredictionResponse,
)
from app.services.ml_service import FraudDetectionService, get_ml_service
from app.db.database import get_db
from app.db import crud
from app.db.schemas import ClaimCreate, PredictionCreate, AuditLogCreate

router = APIRouter()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _claim_to_raw_dict(claim: ClaimInput) -> dict:
    """
    Convert ClaimInput → raw dict with original dataset column names.
    Uses by_alias=True so Days:Policy-Accident, AddressChange-Claim
    are passed with their exact original names to feature_service.
    """
    return claim.model_dump(by_alias=True, exclude_none=True)


def _claim_to_db_schema(claim: ClaimInput) -> ClaimCreate:
    """
    Map ClaimInput API fields → ClaimCreate DB schema.
    Handles the colon/hyphen field aliases cleanly.
    """
    raw = claim.model_dump(by_alias=True, exclude_none=True)
    return ClaimCreate(
        month                   = raw.get("Month"),
        week_of_month           = raw.get("WeekOfMonth"),
        day_of_week             = raw.get("DayOfWeek"),
        month_claimed           = raw.get("MonthClaimed"),
        week_of_month_claimed   = raw.get("WeekOfMonthClaimed"),
        day_of_week_claimed     = raw.get("DayOfWeekClaimed"),
        sex                     = raw.get("Sex"),
        marital_status          = raw.get("MaritalStatus"),
        age                     = raw.get("Age"),
        driver_rating           = raw.get("DriverRating"),
        make                    = raw.get("Make"),
        vehicle_category        = raw.get("VehicleCategory"),
        vehicle_price           = raw.get("VehiclePrice"),
        age_of_vehicle          = raw.get("AgeOfVehicle"),
        age_of_policy_holder    = raw.get("AgeOfPolicyHolder"),
        policy_type             = raw.get("PolicyType"),
        base_policy             = raw.get("BasePolicy"),
        deductible              = raw.get("Deductible"),
        accident_area           = raw.get("AccidentArea"),
        fault                   = raw.get("Fault"),
        agent_type              = raw.get("AgentType"),
        police_report_filed     = raw.get("PoliceReportFiled"),
        witness_present         = raw.get("WitnessPresent"),
        past_number_of_claims   = raw.get("PastNumberOfClaims"),
        number_of_suppliments   = raw.get("NumberOfSuppliments"),
        number_of_cars          = raw.get("NumberOfCars"),
        days_policy_accident    = raw.get("Days:Policy-Accident"),
        days_policy_claim       = raw.get("Days:Policy-Claim"),
        address_change_claim    = raw.get("AddressChange-Claim"),
    )


def _result_to_prediction_schema(
    result: dict,
    claim_id: str,
) -> PredictionCreate:
    """Map the ML service result dict → PredictionCreate DB schema."""
    model_info = result.get("model_info", {})
    return PredictionCreate(
        claim_id            = claim_id,
        is_fraud            = result["is_fraud"],
        fraud_probability   = result["fraud_probability"],
        risk_score          = result["risk_score"],
        risk_level          = result["risk_level"],
        confidence          = result["confidence"],
        recommendation      = result["recommendation"],
        model_name          = model_info.get("model_name", "unknown"),
        model_version       = model_info.get("model_version", "unknown"),
        deployed_threshold  = model_info.get("deployed_threshold", 0.0),
        algorithm           = model_info.get("algorithm", "unknown"),
        inference_ms        = result.get("inference_ms", 0.0),
        risk_factors        = result.get("risk_factors", []),
    )


# ── Single prediction ──────────────────────────────────────────────────────────

@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a single insurance claim",
    description=(
        "Submit claim details and receive an instant fraud risk assessment. "
        "Returns fraud probability, risk score (0-100), risk level, "
        "recommendation, and top 5 contributing risk factors. "
        "Every prediction is saved to the database for audit and analytics."
    ),
    tags=["Predictions"],
)
async def predict_single(
    request : Request,
    claim   : ClaimInput,
    service : FraudDetectionService = Depends(get_ml_service),
    db      : AsyncSession           = Depends(get_db),
):
    """
    Primary prediction endpoint — used by the investigator dashboard.

    Flow:
      1. Validate claim input (Pydantic)
      2. Run ML inference (<100ms)
      3. Save claim + prediction to PostgreSQL
      4. Write audit log entry
      5. Return prediction response
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Check /health endpoint.",
        )

    try:
        # ── Step 1: Run ML inference ──────────────────────────────
        raw_claim = _claim_to_raw_dict(claim)
        result    = service.predict_claim(raw_claim)

        # ── Step 2: Save claim to DB ──────────────────────────────
        claim_schema = _claim_to_db_schema(claim)
        db_claim     = await crud.create_claim(db, claim_schema)

        # ── Step 3: Save prediction to DB ────────────────────────
        pred_schema  = _result_to_prediction_schema(result, db_claim.id)
        db_pred      = await crud.create_prediction(db, pred_schema)

        # ── Step 4: Write audit log ───────────────────────────────
        await crud.log_audit(
            db,
            AuditLogCreate(
                claim_id   = db_claim.id,
                action     = "prediction_created",
                ip_address = request.client.host if request.client else None,
                user_agent = request.headers.get("user-agent"),
                details    = {
                    "prediction_id"    : db_pred.id,
                    "is_fraud"         : result["is_fraud"],
                    "fraud_probability": result["fraud_probability"],
                    "risk_level"       : result["risk_level"],
                    "model_version"    : result.get("model_info", {}).get("model_version"),
                },
            ),
        )

        logger.info(
            f"POST /api/v1/predict | "
            f"claim={db_claim.id} | "
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


# ── Batch prediction ───────────────────────────────────────────────────────────

@router.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse multiple claims in one request",
    description="Submit up to 100 claims for batch fraud analysis. Each is saved to the database.",
    tags=["Predictions"],
)
async def predict_batch(
    request : Request,
    claims  : list[ClaimInput],
    service : FraudDetectionService = Depends(get_ml_service),
    db      : AsyncSession           = Depends(get_db),
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

        # Save each claim + prediction to DB
        for i, claim in enumerate(claims):
            try:
                claim_schema = _claim_to_db_schema(claim)
                db_claim     = await crud.create_claim(db, claim_schema)

                # Grab this claim's individual result from the batch response
                individual   = result["predictions"][i]
                pred_schema  = _result_to_prediction_schema(individual, db_claim.id)
                db_pred      = await crud.create_prediction(db, pred_schema)

                await crud.log_audit(
                    db,
                    AuditLogCreate(
                        claim_id   = db_claim.id,
                        action     = "batch_prediction",
                        ip_address = request.client.host if request.client else None,
                        details    = {
                            "prediction_id": db_pred.id,
                            "batch_index"  : i,
                            "is_fraud"     : individual["is_fraud"],
                        },
                    ),
                )
            except Exception as inner_e:
                # Don't fail the whole batch for a DB write error
                logger.warning(f"DB write failed for batch item {i}: {inner_e}")

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


# ── Recent predictions (from DB) ───────────────────────────────────────────────

@router.get(
    "/api/v1/predictions",
    summary="Get recent predictions from database",
    tags=["Analytics"],
)
async def get_recent_predictions(
    limit      : int  = 50,
    offset     : int  = 0,
    fraud_only : bool = False,
    db         : AsyncSession = Depends(get_db),
):
    """
    Return recent predictions stored in PostgreSQL.
    Used by the claims history dashboard.
    """
    predictions = await crud.list_recent_predictions(
        db,
        limit      = min(limit, 100),   # cap at 100
        offset     = offset,
        fraud_only = fraud_only,
    )
    return {
        "count"      : len(predictions),
        "predictions": [
            {
                "id"               : p.id,
                "claim_id"         : p.claim_id,
                "is_fraud"         : p.is_fraud,
                "fraud_probability": p.fraud_probability,
                "risk_score"       : p.risk_score,
                "risk_level"       : p.risk_level,
                "recommendation"   : p.recommendation,
                "created_at"       : p.created_at.isoformat(),
            }
            for p in predictions
        ],
    }


# ── Analytics summary (from DB) ────────────────────────────────────────────────

@router.get(
    "/api/v1/stats",
    summary="Get fraud detection analytics summary",
    tags=["Analytics"],
)
async def get_stats(
    days : int = 30,
    db   : AsyncSession = Depends(get_db),
):
    """
    Return aggregate fraud statistics for the analytics dashboard.
    Covers the last `days` days (default: 30).
    """
    stats = await crud.get_prediction_stats(db, days=days)
    trend = await crud.get_daily_fraud_trend(db, days=days)
    return {
        "summary": stats,
        "daily_trend": trend,
    }


# ── Model info ─────────────────────────────────────────────────────────────────

@router.get(
    "/api/v1/model/info",
    summary="Get model metadata and performance metrics",
    tags=["Model"],
)
async def model_info(
    service: FraudDetectionService = Depends(get_ml_service),
):
    """Returns model version, training date, test set metrics, threshold in use."""
    return service.get_model_info()


@router.get(
    "/api/v1/model/features",
    summary="Get list of features used by the model",
    tags=["Model"],
)
async def model_features(
    service: FraudDetectionService = Depends(get_ml_service),
):
    """Returns all 76 feature names and their metadata."""
    return service.get_feature_list()