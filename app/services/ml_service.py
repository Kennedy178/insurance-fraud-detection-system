# app/services/ml_service.py
"""
ML Service — Day 16
====================
FraudDetectionService is the single interface between FastAPI endpoints
and the ML model. Endpoints never touch the model directly — they call
this service only.

Architecture:
  FastAPI endpoint
      → FraudDetectionService.predict_claim()
          → FeatureService.transform()   (raw dict → 76 features)
          → FraudDetector.predict()      (ml_pipeline/inference.py)
          → returns PredictionResponse-ready dict

Instantiated once at startup in app/main.py, injected into endpoints
via FastAPI dependency injection.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

# Make ml_pipeline importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
from app.services.feature_service import FeatureService


class FraudDetectionService:
    """
    Production ML service — wraps FraudDetector + FeatureService.

    Responsibilities:
    - Load and hold model in memory (via FraudDetector)
    - Transform raw API claims to model features (via FeatureService)
    - Run inference and return formatted responses
    - Log every prediction to the audit log
    - Track inference timing

    Usage (in main.py):
        fraud_service = FraudDetectionService()

    Usage (in endpoints via DI):
        def predict(service: FraudDetectionService = Depends(get_ml_service)):
            result = service.predict_claim(claim_dict)
    """

    def __init__(self):
        self.detector        = None
        self.feature_service = None
        self._ready          = False
        self._load()

    def _load(self):
        """Load model and feature service. Called once at startup."""
        try:
            from ml_pipeline.inference import FraudDetector

            self.feature_service = FeatureService()
            self.detector        = FraudDetector(model_dir=settings.MODEL_DIR)
            self._ready          = True

            logger.info(
                f"FraudDetectionService ready | "
                f"model={self.detector.metadata['model_name']} | "
                f"threshold={self.detector.deployed_threshold:.4f} | "
                f"features={self.detector.metadata['n_features']}"
            )

        except Exception as e:
            logger.error(f"FraudDetectionService failed to load: {e}")
            self._ready = False
            raise

    # ──────────────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE
    # ──────────────────────────────────────────────────────────────────────

    def predict_claim(self, raw_claim: Dict) -> Dict:
        """
        Full pipeline: raw claim dict → formatted prediction response.

        Args:
            raw_claim: dict with original field names from API request
                       (e.g. {"Fault": "Policy Holder", "AgentType": "External", ...})

        Returns:
            Prediction response dict matching PredictionResponse schema
        """
        self._check_ready()
        start = time.perf_counter()

        # Step 1 — feature engineering (raw → 76 encoded features)
        try:
            X = self.feature_service.transform(raw_claim)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e} | claim={raw_claim}")
            raise ValueError(f"Feature engineering error: {e}")

        # Step 2 — inference (FraudDetector takes a dict, not DataFrame)
        # Pass the processed features as a dict — inference.py's preprocess_input
        # will align columns (they're already aligned, so it's a no-op effectively)
        feature_dict = X.iloc[0].to_dict()

        try:
            result = self.detector.predict(feature_dict)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise ValueError(f"Inference error: {e}")

        total_ms = (time.perf_counter() - start) * 1000
        result['inference_ms'] = round(total_ms, 2)

        # Step 3 — audit log (separate log file via loguru filter)
        logger.bind(prediction=True).info(
            f"PREDICTION | "
            f"fraud={result['is_fraud']} | "
            f"probability={result['fraud_probability']:.4f} | "
            f"risk_level={result['risk_level']} | "
            f"inference_ms={result['inference_ms']}"
        )

        return result

    def predict_batch(self, raw_claims: List[Dict]) -> Dict:
        """
        Batch prediction for POST /api/v1/predict/batch.

        Args:
            raw_claims: list of raw claim dicts

        Returns:
            BatchPredictionResponse-ready dict
        """
        self._check_ready()
        start = time.perf_counter()

        predictions = []
        for claim in raw_claims:
            try:
                result = self.predict_claim(claim)
                predictions.append(result)
            except Exception as e:
                logger.error(f"Batch item failed: {e}")
                # Include error entry rather than failing entire batch
                predictions.append({
                    "error": str(e),
                    "is_fraud": False,
                    "risk_score": 0,
                    "risk_level": "LOW",
                    "fraud_probability": 0.0,
                    "confidence": "low",
                    "recommendation": "Error processing this claim.",
                    "risk_factors": [],
                    "model_info": {
                    "model_name": "fraud_detector_v1",
                    "model_version": "1.0.0",
                    "deployed_threshold": 0.3517,
                    "algorithm": "XGBoost (XGBClassifier)"
                },
                    "inference_ms": 0.0,
                })

        total_ms   = (time.perf_counter() - start) * 1000
        fraud_count = sum(1 for p in predictions if p.get('is_fraud') is True)
        fraud_rate  = fraud_count / len(predictions) if predictions else 0.0

        logger.bind(prediction=True).info(
            f"BATCH_PREDICTION | "
            f"total={len(predictions)} | "
            f"fraud_count={fraud_count} | "
            f"fraud_rate={fraud_rate:.3f} | "
            f"total_ms={total_ms:.1f}"
        )

        return {
            "total_claims" : len(predictions),
            "fraud_count"  : fraud_count,
            "fraud_rate"   : round(fraud_rate, 4),
            "predictions"  : predictions,
            "total_ms"     : round(total_ms, 2),
        }

    # ──────────────────────────────────────────────────────────────────────
    # MODEL INFO
    # ──────────────────────────────────────────────────────────────────────

    def get_model_info(self) -> Dict:
        """For GET /api/v1/model/info endpoint (Day 17)"""
        self._check_ready()
        return self.detector.get_model_info()

    def get_feature_list(self) -> Dict:
        """For GET /api/v1/model/features endpoint (Day 17)"""
        self._check_ready()
        return {
            "n_features"   : self.detector.metadata['n_features'],
            "feature_names": self.detector.feature_names,
            "feature_details": self.detector.metadata.get('feature_details', []),
        }

    def is_ready(self) -> bool:
        return self._ready

    def _check_ready(self):
        if not self._ready:
            raise RuntimeError(
                "FraudDetectionService not ready — model failed to load at startup. "
                "Check logs for details."
            )


# ── Dependency injection helper for FastAPI ───────────────────────────────
# Used in endpoint files:
#   from app.services.ml_service import get_ml_service
#   def endpoint(service = Depends(get_ml_service)): ...

_service_instance: Optional[FraudDetectionService] = None


def get_ml_service() -> FraudDetectionService:
    """
    FastAPI dependency — returns the singleton service instance.
    Initialized in main.py lifespan, injected into endpoints via Depends().
    """
    global _service_instance
    if _service_instance is None:
        raise RuntimeError(
            "ML service not initialized. "
            "Ensure init_ml_service() is called in app lifespan."
        )
    return _service_instance


def init_ml_service() -> FraudDetectionService:
    """Called once from main.py lifespan startup"""
    global _service_instance
    _service_instance = FraudDetectionService()
    return _service_instance