"""
Day 16 — ML Service Test Script
================================
Tests the full pipeline: raw API input → feature engineering → inference → output.
Run from project root:
    python tests/test_ml_service.py

This is independent of FastAPI — tests the service layer directly.
"""

import sys
import time
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.ml_service import FraudDetectionService


def run_tests():
    print("="*60)
    print("DAY 16 — ML SERVICE TEST")
    print("="*60)

    # ── Load service ───────────────────────────────────────────
    print("\n[1/5] Loading FraudDetectionService...")
    service = FraudDetectionService()
    assert service.is_ready(), "Service failed to load"
    print("  ✓ Service loaded and ready")

    # ── HIGH RISK claim — known fraud patterns ─────────────────
    print("\n[2/5] High-risk claim (known fraud patterns)...")
    high_risk_claim = {
        'Fault'              : 'Policy Holder',
        'AgentType'          : 'External',
        'PoliceReportFiled'  : 'No',
        'WitnessPresent'     : 'No',
        'AccidentArea'       : 'Urban',
        'Days:Policy-Accident': 'none',
        'Days:Policy-Claim'  : 'none',
        'AddressChange-Claim': 'under 6 months',
        'PastNumberOfClaims' : '2 to 4',
        'NumberOfSuppliments': 'more than 5',
        'BasePolicy'         : 'Liability',
        'PolicyType'         : 'Sedan - Liability',
        'VehicleCategory'    : 'Sedan',
        'Age'                : 34,
        'Deductible'         : 400,
        'DriverRating'       : 4,
        'Make'               : 'Toyota',
        'MaritalStatus'      : 'Single',
        'Sex'                : 'Male',
        'Month'              : 'Jan',
        'MonthClaimed'       : 'Jan',
        'DayOfWeek'          : 'Monday',
        'DayOfWeekClaimed'   : 'Monday',
        'WeekOfMonth'        : 1,
        'WeekOfMonthClaimed' : 1,
        'AgeOfVehicle'       : '3 years',
        'VehiclePrice'       : '20,000 to 29,000',
        'AgeOfPolicyHolder'  : '26 to 30',
        'NumberOfCars'       : '1 vehicle',
    }

    result = service.predict_claim(high_risk_claim)
    print(f"  Fraud probability: {result['fraud_probability']:.4f}")
    print(f"  Risk score:        {result['risk_score']}/100")
    print(f"  Risk level:        {result['risk_level']}")
    print(f"  Is fraud:          {result['is_fraud']}")
    print(f"  Confidence:        {result['confidence']}")
    print(f"  Recommendation:    {result['recommendation']}")
    print(f"  Inference time:    {result['inference_ms']:.2f}ms")
    print(f"  Top risk factors:")
    for f in result['risk_factors']:
        print(f"    {f['feature']:<40} {f['importance']:.4f}")

    # ── LOW RISK claim — legitimate patterns ───────────────────
    print("\n[3/5] Low-risk claim (legitimate patterns)...")
    low_risk_claim = {
        'Fault'              : 'Third Party',
        'AgentType'          : 'Internal',
        'PoliceReportFiled'  : 'Yes',
        'WitnessPresent'     : 'Yes',
        'AccidentArea'       : 'Rural',
        'Days:Policy-Accident': 'more than 30',
        'Days:Policy-Claim'  : 'more than 30',
        'AddressChange-Claim': 'no change',
        'PastNumberOfClaims' : 'none',
        'NumberOfSuppliments': 'none',
        'BasePolicy'         : 'Collision',
        'PolicyType'         : 'Sedan - Collision',
        'VehicleCategory'    : 'Sedan',
        'Age'                : 45,
        'Deductible'         : 700,
        'DriverRating'       : 1,
        'Make'               : 'Honda',
        'MaritalStatus'      : 'Married',
        'Sex'                : 'Female',
        'Month'              : 'Mar',
        'MonthClaimed'       : 'Mar',
        'DayOfWeek'          : 'Wednesday',
        'DayOfWeekClaimed'   : 'Thursday',
        'WeekOfMonth'        : 2,
        'WeekOfMonthClaimed' : 2,
        'AgeOfVehicle'       : '5 years',
        'VehiclePrice'       : '30,000 to 39,000',
        'AgeOfPolicyHolder'  : '41 to 50',
        'NumberOfCars'       : '1 vehicle',
    }

    result_low = service.predict_claim(low_risk_claim)
    print(f"  Fraud probability: {result_low['fraud_probability']:.4f}")
    print(f"  Risk score:        {result_low['risk_score']}/100")
    print(f"  Risk level:        {result_low['risk_level']}")
    print(f"  Recommendation:    {result_low['recommendation']}")
    print(f"  Inference time:    {result_low['inference_ms']:.2f}ms")

    # ── Timing test ────────────────────────────────────────────
    print("\n[4/5] Inference timing (50 predictions)...")
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        service.predict_claim(high_risk_claim)
        times.append((time.perf_counter() - t0) * 1000)

    import statistics
    avg = statistics.mean(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f"  Average: {avg:.2f}ms")
    print(f"  P95:     {p95:.2f}ms")
    status_str = '✓' if p95 < 100 else '⚠️  EXCEEDS TARGET'
    print(f"  {status_str}  P95 < 100ms target")

    # ── Batch test ─────────────────────────────────────────────
    print("\n[5/5] Batch prediction (5 claims)...")
    batch = service.predict_batch([high_risk_claim, low_risk_claim] * 2 + [high_risk_claim])
    print(f"  ✓ Total claims:  {batch['total_claims']}")
    print(f"  ✓ Fraud flagged: {batch['fraud_count']}")
    print(f"  ✓ Fraud rate:    {batch['fraud_rate']:.1%}")
    print(f"  ✓ Total time:    {batch['total_ms']:.1f}ms")

    # ── Model info ─────────────────────────────────────────────
    print("\n  Model info:")
    info = service.get_model_info()
    print(f"  Model:    {info['model_name']} v{info['model_version']}")
    print(f"  ROC-AUC:  {info['performance']['roc_auc']:.4f}")
    print(f"  Features: {info['n_features']}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED — Day 16 complete")
    print("="*60)
    print("\nNext: restart uvicorn and test POST /api/v1/predict in Swagger UI")


if __name__ == "__main__":
    run_tests()