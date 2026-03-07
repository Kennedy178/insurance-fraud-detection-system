# app/models/prediction.py
"""
Pydantic schemas for prediction responses.
These define exactly what the API returns — matches format_output()
in ml_pipeline/inference.py FraudDetector class.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RiskLevelEnum(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class ConfidenceLevelEnum(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"


class RiskFactor(BaseModel):
    """Single feature contribution to the prediction"""
    feature    : str   = Field(..., description="Feature name")
    importance : float = Field(..., description="Feature importance score")
    value      : float = Field(..., description="Feature value for this claim")
    description: str   = Field(..., description="Human-readable explanation")


class ModelInfo(BaseModel):
    """Model metadata returned with every prediction"""
    model_name        : str   = Field(..., description="Model identifier")
    model_version     : str   = Field(..., description="Model version string")
    deployed_threshold: float = Field(..., description="Decision threshold in use")
    algorithm         : str   = Field(..., description="Algorithm name")


class PredictionResponse(BaseModel):
    """
    Full prediction response returned by POST /api/v1/predict.
    Designed for the investigator dashboard — every field maps to
    a UI component.
    """
    is_fraud          : bool              = Field(..., description="True if claim flagged as fraud")
    fraud_probability : float             = Field(..., ge=0.0, le=1.0, description="Raw fraud probability (0-1)")
    risk_score        : int               = Field(..., ge=0, le=100, description="Risk score 0-100 for dashboard display")
    risk_level        : RiskLevelEnum     = Field(..., description="LOW / MEDIUM / HIGH")
    confidence        : ConfidenceLevelEnum = Field(..., description="Model confidence in this prediction")
    recommendation    : str               = Field(..., description="Plain-English action for investigator")
    risk_factors      : List[RiskFactor]  = Field(..., description="Top 5 contributing features")
    model_info        : ModelInfo         = Field(..., description="Model version and threshold used")
    inference_ms      : float             = Field(..., description="Server-side inference time in ms")

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_fraud": True,
                "fraud_probability": 0.9832,
                "risk_score": 98,
                "risk_level": "HIGH",
                "confidence": "high",
                "recommendation": "High fraud probability. Flag for immediate investigation.",
                "risk_factors": [
                    {
                        "feature": "external_agent_holder_fault",
                        "importance": 0.1894,
                        "value": 1.0,
                        "description": "Third-party agent involvement detected"
                    }
                ],
                "model_info": {
                    "model_name": "fraud_detector_v1",
                    "model_version": "1.0.0",
                    "deployed_threshold": 0.3517,
                    "algorithm": "XGBoost (XGBClassifier)"
                },
                "inference_ms": 54.71
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response for POST /api/v1/predict/batch"""
    total_claims  : int                    = Field(..., description="Number of claims processed")
    fraud_count   : int                    = Field(..., description="Number flagged as fraud")
    fraud_rate    : float                  = Field(..., description="Fraud rate in this batch")
    predictions   : List[PredictionResponse] = Field(..., description="Individual prediction results")
    total_ms      : float                  = Field(..., description="Total processing time in ms")


class ModelInfoResponse(BaseModel):
    """
    Response for GET /api/v1/model/info.
    Full model metadata including performance metrics.
    """
    model_name         : str   = Field(..., description="Model identifier")
    model_version      : str   = Field(..., description="Version string")
    algorithm          : str   = Field(..., description="Algorithm")
    n_features         : int   = Field(..., description="Number of input features")
    training_date      : str   = Field(..., description="Date model was trained")
    deployed_threshold : float = Field(..., description="Current decision threshold")
    performance        : dict  = Field(..., description="Test set metrics")
    training_data      : dict  = Field(..., description="Training dataset info")


class HealthResponse(BaseModel):
    """Response for GET /health"""
    status       : str  = Field(..., description="API status: ok | degraded | down")
    version      : str  = Field(..., description="API version")
    model_loaded : bool = Field(..., description="Whether ML model is loaded")
    environment  : str  = Field(..., description="development | production")
    uptime_seconds: Optional[float] = Field(None, description="Seconds since startup")