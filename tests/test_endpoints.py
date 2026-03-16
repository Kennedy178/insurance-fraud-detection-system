"""
Day 17 — Endpoint Hardening Tests
===================================
Tests all 4 core endpoints with valid, invalid, and edge case inputs.
Uses httpx to hit the live FastAPI server — uvicorn must be running.

Run from project root (with uvicorn running in another terminal):
    python tests/test_endpoints.py

Or use pytest:
    pip install httpx pytest
    pytest tests/test_endpoints.py -v
"""

import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Install httpx first: pip install httpx")
    sys.exit(1)

BASE_URL = "http://localhost:8000"

# ── Sample claims ──────────────────────────────────────────────────────────

HIGH_RISK_CLAIM = {
    "fault"              : "Policy Holder",
    "agent_type"         : "External",
    "police_report_filed": "No",
    "witnesses"          : 0,
    "accident_area"      : "Urban",
    "base_policy"        : "Liability",
    "policy_type"        : "Sedan - Liability",
    "vehicle_category"   : "Sedan",
    "age"                : 34,
    "deductible"         : 400,
    "driver_rating"      : 4,
}

LOW_RISK_CLAIM = {
    "fault"              : "Third Party",
    "agent_type"         : "Internal",
    "police_report_filed": "Yes",
    "witnesses"          : 2,
    "accident_area"      : "Rural",
    "base_policy"        : "Collision",
    "policy_type"        : "Sedan - Collision",
    "vehicle_category"   : "Sedan",
    "age"                : 45,
    "deductible"         : 700,
    "driver_rating"      : 1,
}

MINIMAL_CLAIM = {
    "fault": "Policy Holder",
}

EMPTY_CLAIM = {}


def print_result(test_name: str, passed: bool, detail: str = ""):
    icon = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  {icon} [{status}] {test_name}", end="")
    if detail:
        print(f"  — {detail}", end="")
    print()
    return passed


def run_all_tests():
    print("=" * 60)
    print("DAY 17 — ENDPOINT HARDENING TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:

        # ── 1. HEALTH ENDPOINT ─────────────────────────────────
        print("\n[GET /health]")

        r = client.get("/health")
        ok = r.status_code == 200 and r.json()["model_loaded"] is True
        passed += ok; failed += not ok
        print_result("Returns 200 with model_loaded=true", ok, f"status={r.status_code}")

        start = time.perf_counter()
        client.get("/health")
        ms = (time.perf_counter() - start) * 1000
        ok = ms < 200  # generous for localhost overhead
        passed += ok; failed += not ok
        print_result(f"Response time acceptable", ok, f"{ms:.1f}ms")

        ok = "status" in r.json() and "version" in r.json()
        passed += ok; failed += not ok
        print_result("Response schema complete", ok)

        # ── 2. STATUS ENDPOINT ─────────────────────────────────
        print("\n[GET /api/v1/status]")

        r = client.get("/api/v1/status")
        ok = r.status_code == 200
        passed += ok; failed += not ok
        print_result("Returns 200", ok)

        data = r.json()
        ok = (data.get("model", {}).get("loaded") is True and
              data.get("model", {}).get("threshold") is not None)
        passed += ok; failed += not ok
        print_result("Model info present", ok,
                     f"threshold={data.get('model', {}).get('threshold')}")

        # ── 3. POST /api/v1/predict — VALID INPUTS ─────────────
        print("\n[POST /api/v1/predict — valid inputs]")

        # High risk claim
        r = client.post("/api/v1/predict", json=HIGH_RISK_CLAIM)
        ok = r.status_code == 200
        passed += ok; failed += not ok
        data = r.json()
        print_result("High-risk claim returns 200", ok,
                     f"probability={data.get('fraud_probability', '?')}")

        ok = all(k in data for k in [
            "is_fraud", "fraud_probability", "risk_score",
            "risk_level", "confidence", "recommendation",
            "risk_factors", "model_info", "inference_ms"
        ])
        passed += ok; failed += not ok
        print_result("Response has all required fields", ok)

        ok = 0 <= data.get("fraud_probability", -1) <= 1
        passed += ok; failed += not ok
        print_result("fraud_probability in [0,1]", ok)

        ok = 0 <= data.get("risk_score", -1) <= 100
        passed += ok; failed += not ok
        print_result("risk_score in [0,100]", ok)

        ok = data.get("risk_level") in ["LOW", "MEDIUM", "HIGH"]
        passed += ok; failed += not ok
        print_result("risk_level is valid enum", ok,
                     f"got={data.get('risk_level')}")

        ok = len(data.get("risk_factors", [])) > 0
        passed += ok; failed += not ok
        print_result("risk_factors list not empty", ok,
                     f"count={len(data.get('risk_factors', []))}")

        # Low risk claim
        r = client.post("/api/v1/predict", json=LOW_RISK_CLAIM)
        ok = r.status_code == 200
        passed += ok; failed += not ok
        low_data = r.json()
        print_result("Low-risk claim returns 200", ok,
                     f"probability={low_data.get('fraud_probability', '?')}")

        # Verify high risk > low risk (model is discriminating)
        high_prob = data.get("fraud_probability", 0)
        low_prob  = low_data.get("fraud_probability", 1)
        ok = high_prob > low_prob
        passed += ok; failed += not ok
        print_result("High-risk probability > low-risk probability", ok,
                     f"{high_prob:.4f} > {low_prob:.4f}")

        # Minimal claim (only one field)
        r = client.post("/api/v1/predict", json=MINIMAL_CLAIM)
        ok = r.status_code == 200
        passed += ok; failed += not ok
        print_result("Minimal claim (1 field) returns 200", ok)

        # Empty claim
        r = client.post("/api/v1/predict", json=EMPTY_CLAIM)
        ok = r.status_code == 200
        passed += ok; failed += not ok
        print_result("Empty claim returns 200 (graceful defaults)", ok)

        # ── 4. POST /api/v1/predict — INVALID INPUTS ───────────
        print("\n[POST /api/v1/predict — invalid inputs]")

        # Age out of range
        r = client.post("/api/v1/predict", json={"age": 999})
        ok = r.status_code == 422
        passed += ok; failed += not ok
        print_result("Age=999 returns 422 validation error", ok,
                     f"status={r.status_code}")

        # Negative deductible
        r = client.post("/api/v1/predict", json={"deductible": -500})
        ok = r.status_code == 422
        passed += ok; failed += not ok
        print_result("Negative deductible returns 422", ok)

        # Invalid enum value
        r = client.post("/api/v1/predict", json={"fault": "INVALID_VALUE"})
        ok = r.status_code == 422
        passed += ok; failed += not ok
        print_result("Invalid fault enum returns 422", ok)

        # Wrong content type (send string not JSON)
        r = client.post(
            "/api/v1/predict",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        ok = r.status_code == 422
        passed += ok; failed += not ok
        print_result("Malformed JSON returns 422", ok)

        # ── 5. POST /api/v1/predict/batch ──────────────────────
        print("\n[POST /api/v1/predict/batch]")

        batch = [HIGH_RISK_CLAIM, LOW_RISK_CLAIM, MINIMAL_CLAIM]
        r = client.post("/api/v1/predict/batch", json=batch)
        ok = r.status_code == 200
        passed += ok; failed += not ok
        data = r.json()
        print_result("Batch of 3 returns 200", ok)

        ok = data.get("total_claims") == 3
        passed += ok; failed += not ok
        print_result("total_claims=3", ok)

        ok = len(data.get("predictions", [])) == 3
        passed += ok; failed += not ok
        print_result("predictions array has 3 items", ok)

        # Batch too large
        big_batch = [HIGH_RISK_CLAIM] * 101
        r = client.post("/api/v1/predict/batch", json=big_batch)
        ok = r.status_code == 400
        passed += ok; failed += not ok
        print_result("Batch >100 returns 400", ok)

        # Empty batch
        r = client.post("/api/v1/predict/batch", json=[])
        ok = r.status_code == 400
        passed += ok; failed += not ok
        print_result("Empty batch returns 400", ok)

        # ── 6. MODEL INFO ENDPOINTS ─────────────────────────────
        print("\n[GET /api/v1/model/info and /features]")

        r = client.get("/api/v1/model/info")
        ok = r.status_code == 200
        passed += ok; failed += not ok
        info = r.json()
        print_result("GET /model/info returns 200", ok)

        ok = all(k in info for k in ["model_name", "model_version",
                                      "algorithm", "n_features", "performance"])
        passed += ok; failed += not ok
        print_result("Model info has all required fields", ok)

        r = client.get("/api/v1/model/features")
        ok = r.status_code == 200
        passed += ok; failed += not ok
        features = r.json()
        print_result("GET /model/features returns 200", ok)

        ok = features.get("n_features") == 76
        passed += ok; failed += not ok
        print_result("n_features=76", ok,
                     f"got={features.get('n_features')}")

        # ── 7. OHE FIX VERIFICATION ─────────────────────────────
        print("\n[OHE fix verification — BasePolicy_Liability should be 1]")

        r = client.post("/api/v1/predict", json={
            "base_policy"    : "Liability",
            "policy_type"    : "Sedan - Liability",
            "vehicle_category": "Sedan",
            "fault"          : "Policy Holder",
            "agent_type"     : "External",
        })
        ok = r.status_code == 200
        passed += ok; failed += not ok
        data = r.json()

        # Find BasePolicy_Liability in risk factors
        bp_liability = next(
            (f for f in data.get("risk_factors", [])
             if f["feature"] == "BasePolicy_Liability"), None
        )
        if bp_liability:
            ok = bp_liability["value"] == 1.0
            passed += ok; failed += not ok
            print_result(
                "BasePolicy_Liability value=1 when base_policy=Liability",
                ok, f"value={bp_liability['value']}"
            )
        else:
            # Not in top 5 risk factors — check probability is higher than before
            ok = data.get("fraud_probability", 0) > 0.20
            passed += ok; failed += not ok
            print_result(
                "Probability elevated for Liability+External+Fault combo",
                ok, f"probability={data.get('fraud_probability')}"
            )

    # ── SUMMARY ────────────────────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED — Day 17 complete")
    else:
        print(f"⚠️  {failed} test(s) failed — check output above")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    print(f"Testing API at {BASE_URL}")
    print("Make sure uvicorn is running: uvicorn app.main:app --reload")
    print()
    success = run_all_tests()
    sys.exit(0 if success else 1)