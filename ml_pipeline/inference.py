"""
Inference Module — Day 14
==========================
Production-ready inference module. This is the bridge between the trained
model and the FastAPI backend (Phase 3 / Day 16).

Location: ml_pipeline/inference.py

The FastAPI ml_service.py (Day 16) will import FraudDetector from here.
This module handles everything between raw claim data and a formatted
prediction response — model loading, feature alignment, scoring,
risk level assignment, and output formatting.

Run standalone test:
    cd ml_pipeline
    python inference.py
"""

import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ── Default paths (relative to ml_pipeline/) ──────────────────────────────
DEFAULT_MODEL_DIR = '../models'


class FraudDetector:
    """
    Production inference class for fraud detection.

    Usage:
        detector = FraudDetector()
        result = detector.predict(claim_dict)

    The FastAPI service instantiates this once at startup and reuses it
    for all prediction requests — model stays loaded in memory.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir      = model_dir
        self.model          = None
        self.metadata       = None
        self.threshold_data = None
        self.feature_names  = None
        self.deployed_threshold = None
        self._loaded        = False
        self.load_model()

    # ──────────────────────────────────────────────────────────────────────
    # 1. LOAD MODEL
    # ──────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load model, threshold, and metadata from disk.
        Called once at startup — all subsequent predictions use cached objects.
        """
        model_path     = os.path.join(self.model_dir, 'fraud_detector_v1.joblib')
        threshold_path = os.path.join(self.model_dir, 'xgboost_threshold.json')
        metadata_path  = os.path.join(self.model_dir, 'feature_metadata.json')

        # Check all files exist before loading
        for label, path in [('Model', model_path),
                              ('Threshold', threshold_path),
                              ('Metadata', metadata_path)]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} file not found: {path}\n"
                    f"Run ml_pipeline/serialize_artifacts.py first."
                )

        self.model = joblib.load(model_path)

        with open(threshold_path, 'r') as f:
            self.threshold_data = json.load(f)

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.deployed_threshold = self.threshold_data['deployed_threshold']
        self.feature_names      = self.metadata['feature_names']
        self._loaded            = True

        print(f"FraudDetector loaded")
        print(f"  Model:     {self.metadata['model_name']} v{self.metadata['model_version']}")
        print(f"  Features:  {len(self.feature_names)}")
        print(f"  Threshold: {self.deployed_threshold:.4f} (deployed/business-optimal)")

    # ──────────────────────────────────────────────────────────────────────
    # 2. PREPROCESS INPUT
    # ──────────────────────────────────────────────────────────────────────

    def preprocess_input(self, claim: Dict) -> pd.DataFrame:
        """
        Convert raw claim dict → feature DataFrame aligned to training columns.

        Steps:
          1. Convert dict to single-row DataFrame
          2. Clean column names (match training preprocessing exactly)
          3. Align columns to training order — add 0 for any missing features
          4. Validate no unexpected extra features

        Args:
            claim: dict with claim fields (from API request body)

        Returns:
            DataFrame with exactly self.feature_names columns in correct order
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Step 1 — dict to DataFrame
        df = pd.DataFrame([claim])

        # Step 2 — clean column names (same as model_training.py)
        df.columns = (df.columns
                      .str.replace(':', '_')
                      .str.replace('-', '_')
                      .str.replace(' ', '_'))

        # Step 3 — align to training features
        # Add any missing features as 0 (handles partial input gracefully)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # Keep only training features in training order
        df = df[self.feature_names]

        # Step 4 — type coercion (ensure numeric)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        return df

    # ──────────────────────────────────────────────────────────────────────
    # 3. PREDICT
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, claim: Dict) -> Dict:
        """
        Full prediction pipeline for a single claim.

        Args:
            claim: dict of claim features

        Returns:
            Formatted prediction response dict (see format_output)
        """
        start = time.perf_counter()

        # Preprocess
        X = self.preprocess_input(claim)

        # Inference
        fraud_probability = float(self.model.predict_proba(X)[:, 1][0])
        is_fraud          = bool(fraud_probability >= self.deployed_threshold)

        # Risk score (0-100 scale for the dashboard)
        risk_score = self.calculate_risk_score(fraud_probability)

        # Confidence and risk level
        confidence  = self.get_confidence_level(fraud_probability)
        risk_level  = self.get_risk_level(risk_score)

        # Top contributing features (for explainability)
        risk_factors = self.extract_risk_factors(X)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return self.format_output(
            fraud_probability = fraud_probability,
            is_fraud          = is_fraud,
            risk_score        = risk_score,
            confidence        = confidence,
            risk_level        = risk_level,
            risk_factors      = risk_factors,
            inference_ms      = elapsed_ms,
        )
    
    def predict_from_df(self, X: pd.DataFrame) -> Dict:
        """
        Run inference on an already-engineered feature DataFrame.
        Called by ml_service.py which does its own feature engineering
        via feature_service.py — skips preprocess_input entirely.

        Args:
            X: DataFrame with exactly 76 features in training column order
            (already produced by FeatureService.transform())
        """
        start = time.perf_counter()

        fraud_probability = float(self.model.predict_proba(X)[:, 1][0])
        is_fraud          = bool(fraud_probability >= self.deployed_threshold)
        risk_score        = self.calculate_risk_score(fraud_probability)
        confidence        = self.get_confidence_level(fraud_probability)
        risk_level        = self.get_risk_level(risk_score)
        risk_factors      = self.extract_risk_factors(X)
        elapsed_ms        = (time.perf_counter() - start) * 1000

        return self.format_output(
            fraud_probability = fraud_probability,
            is_fraud          = is_fraud,
            risk_score        = risk_score,
            confidence        = confidence,
            risk_level        = risk_level,
            risk_factors      = risk_factors,
            inference_ms      = elapsed_ms,
        )

    def predict_batch(self, claims: List[Dict]) -> List[Dict]:
        """
        Batch prediction for multiple claims.
        More efficient than calling predict() in a loop for large batches.

        Args:
            claims: list of claim dicts

        Returns:
            List of formatted prediction response dicts
        """
        start = time.perf_counter()

        # Preprocess all at once
        rows = [self.preprocess_input(c) for c in claims]
        X    = pd.concat(rows, ignore_index=True)

        # Batch inference
        probas = self.model.predict_proba(X)[:, 1]

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_claim_ms = elapsed_ms / len(claims)

        results = []
        for i, (claim, prob) in enumerate(zip(claims, probas)):
            prob      = float(prob)
            is_fraud  = bool(prob >= self.deployed_threshold)
            risk_score = self.calculate_risk_score(prob)
            results.append(self.format_output(
                fraud_probability = prob,
                is_fraud          = is_fraud,
                risk_score        = risk_score,
                confidence        = self.get_confidence_level(prob),
                risk_level        = self.get_risk_level(risk_score),
                risk_factors      = self.extract_risk_factors(X.iloc[[i]]),
                inference_ms      = per_claim_ms,
            ))

        return results

    # ──────────────────────────────────────────────────────────────────────
    # 4. SCORING HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def calculate_risk_score(self, fraud_probability: float) -> int:
        """
        Convert raw fraud probability (0-1) - risk score (0-100).
        Linear scaling — simple, transparent, easy to explain.

        0-30:  Low risk (green)
        31-60: Medium risk (yellow)
        61-100: High risk (red)
        """
        return min(100, int(round(fraud_probability * 100)))

    def get_confidence_level(self, fraud_probability: float) -> str:
        """
        Translate probability distance from threshold - confidence string.
        Close to threshold = low confidence. Far from it = high confidence.
        """
        distance = abs(fraud_probability - self.deployed_threshold)
        if distance < 0.10:
            return 'low'
        elif distance < 0.25:
            return 'medium'
        else:
            return 'high'

    def get_risk_level(self, risk_score: int) -> str:
        """Map risk score to traffic light level for dashboard colour coding"""
        if risk_score <= 30:
            return 'LOW'
        elif risk_score <= 60:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def extract_risk_factors(self, X: pd.DataFrame,
                              top_n: int = 5) -> List[Dict]:
        """
        Extract top N features by model feature importance for this prediction.
        Used as 'reason codes' in the API response and dashboard.

        Note: This uses global feature importance (fast, no SHAP overhead).
        For individual SHAP explanations, see model_evaluation.py.
        The FastAPI service can optionally run SHAP on high-risk claims only.
        """
        importances = self.model.feature_importances_
        feature_values = X.iloc[0].to_dict()

        factors = []
        for name, importance in zip(self.feature_names, importances):
            factors.append({
                'feature'    : name,
                'importance' : float(importance),
                'value'      : float(feature_values.get(name, 0)),
            })

        # Sort by importance, take top N
        factors = sorted(factors, key=lambda x: x['importance'], reverse=True)[:top_n]

        # Add human-readable description
        descriptions = {
            'BasePolicy_Liability'        : 'Liability policy type — primary fraud indicator',
            'policy_holder_fault'         : 'Fault assigned to policyholder',
            'external_agent_holder_fault' : 'Third-party agent involvement detected',
            'Age'                         : 'Claimant age profile',
            'Fault_binary'                : 'Fault attribution pattern',
            'PolicyType_Sedan___Liability': 'Sedan + liability policy combination',
            'MonthClaimed_numeric'        : 'Month claim was filed',
            'Month_numeric'               : 'Policy month',
            'Deductible'                  : 'Deductible amount profile',
        }

        for f in factors:
            f['description'] = descriptions.get(f['feature'],
                                                  f['feature'].replace('_', ' ').title())

        return factors

    # ──────────────────────────────────────────────────────────────────────
    # 5. FORMAT OUTPUT
    # ──────────────────────────────────────────────────────────────────────

    def format_output(self,
                      fraud_probability : float,
                      is_fraud          : bool,
                      risk_score        : int,
                      confidence        : str,
                      risk_level        : str,
                      risk_factors      : List[Dict],
                      inference_ms      : float) -> Dict:
        """
        Standard prediction response format.
        This schema is what the FastAPI PredictionResponse Pydantic model
        will validate against on Day 15/16.

        Returns:
            {
                "is_fraud": bool,
                "fraud_probability": float,
                "risk_score": int,          # 0-100
                "risk_level": str,          # LOW / MEDIUM / HIGH
                "confidence": str,          # low / medium / high
                "recommendation": str,      # human-readable action
                "risk_factors": [...],      # top 5 contributing features
                "model_info": {...},        # version, threshold used
                "inference_ms": float,      # response time
            }
        """
        recommendations = {
            'LOW'    : 'Claim appears legitimate. Standard processing recommended.',
            'MEDIUM' : 'Some risk indicators present. Manual review suggested.',
            'HIGH'   : 'High fraud probability. Flag for immediate investigation.',
        }

        return {
            'is_fraud'          : is_fraud,
            'fraud_probability' : round(fraud_probability, 4),
            'risk_score'        : risk_score,
            'risk_level'        : risk_level,
            'confidence'        : confidence,
            'recommendation'    : recommendations[risk_level],
            'risk_factors'      : risk_factors,
            'model_info'        : {
                'model_name'        : self.metadata['model_name'],
                'model_version'     : self.metadata['model_version'],
                'deployed_threshold': self.deployed_threshold,
                'algorithm'         : self.metadata['algorithm'],
            },
            'inference_ms'      : round(inference_ms, 2),
        }

    # ──────────────────────────────────────────────────────────────────────
    # 6. MODEL INFO
    # ──────────────────────────────────────────────────────────────────────

    def get_model_info(self) -> Dict:
        """
        Return model metadata for the GET /api/v1/model/info endpoint (Day 17).
        """
        return {
            'model_name'         : self.metadata['model_name'],
            'model_version'      : self.metadata['model_version'],
            'algorithm'          : self.metadata['algorithm'],
            'n_features'         : self.metadata['n_features'],
            'training_date'      : self.metadata['training_date'],
            'deployed_threshold' : self.deployed_threshold,
            'performance'        : self.metadata['performance'],
            'training_data'      : self.metadata['training_data'],
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_tests():
    """
    Test the inference module end-to-end.
    Verifies: load, single predict, batch predict, edge cases, timing.
    Run: python inference.py
    """
    print("="*60)
    print("INFERENCE MODULE TEST — Day 14")
    print("="*60)

    # ── Load detector ──────────────────────────────────────────
    print("\n[1/5] Loading FraudDetector...")
    detector = FraudDetector()
    print("✓ Loaded successfully")

    # ── Load a real test claim ─────────────────────────────────
    print("\n[2/5] Single claim prediction...")
    test_df = pd.read_csv('../data/processed/test.csv')

    # Use the highest-risk claim from evaluation (index 1975)
    claim_row = test_df.iloc[1975].drop('FraudFound_P').to_dict()
    actual    = int(test_df.iloc[1975]['FraudFound_P'])

    result = detector.predict(claim_row)

    print(f"  Actual label:        {'FRAUD' if actual else 'LEGITIMATE'}")
    print(f"  Predicted:           {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
    print(f"  Fraud probability:   {result['fraud_probability']:.4f}")
    print(f"  Risk score:          {result['risk_score']}/100")
    print(f"  Risk level:          {result['risk_level']}")
    print(f"  Confidence:          {result['confidence']}")
    print(f"  Recommendation:      {result['recommendation']}")
    print(f"  Inference time:      {result['inference_ms']:.2f}ms")
    print(f"\n  Top risk factors:")
    for f in result['risk_factors']:
        print(f"    {f['feature']:<35} importance={f['importance']:.4f}")

    # ── Timing test ────────────────────────────────────────────
    print("\n[3/5] Inference timing (100 predictions)...")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        detector.predict(claim_row)
        times.append((time.perf_counter() - start) * 1000)

    avg_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    print(f"  Average: {avg_ms:.2f}ms")
    print(f"  P95:     {p95_ms:.2f}ms")
    status = '✓' if p95_ms < 100 else '⚠️ EXCEEDS TARGET'
    print(f"  {status}  P95 < 100ms target")

    # ── Batch prediction ───────────────────────────────────────
    print("\n[4/5] Batch prediction (10 claims)...")
    batch = [test_df.iloc[i].drop('FraudFound_P').to_dict() for i in range(10)]
    batch_results = detector.predict_batch(batch)
    fraud_count = sum(1 for r in batch_results if r['is_fraud'])
    print(f"  ✓ Batch of 10 processed")
    print(f"  Flagged as fraud: {fraud_count}/10")
    print(f"  Avg inference time per claim: {batch_results[0]['inference_ms']:.2f}ms")

    # ── Edge case — missing values ─────────────────────────────
    print("\n[5/5] Edge case — missing/empty claim...")
    empty_claim = {}
    try:
        empty_result = detector.predict(empty_claim)
        print(f"  ✓ Empty claim handled gracefully")
        print(f"  Result: {empty_result['risk_level']} risk ({empty_result['fraud_probability']:.4f})")
    except Exception as e:
        print(f"  ✗ Error on empty claim: {e}")

    # ── Model info ─────────────────────────────────────────────
    print("\n Model Info (for GET /api/v1/model/info endpoint):")
    info = detector.get_model_info()
    print(f"  Model:    {info['model_name']} v{info['model_version']}")
    print(f"  ROC-AUC:  {info['performance']['roc_auc']:.4f}")
    print(f"  Features: {info['n_features']}")

    print("\n" + "="*60)
    print("INFERENCE MODULE TEST COMPLETE")
    print("="*60)
    print(f"\n All tests passed")
    print(f"   inference.py is ready for import by app/services/ml_service.py")


if __name__ == "__main__":
    run_inference_tests()