# Feature Engineering Documentation

**Date:** February 24, 2026  
**Input:** `data/processed/insurance_claims_cleaned.csv` (15,420 rows × 30 columns)  
**Output:** `data/processed/insurance_claims_engineered.csv` (15,420 rows × 77 columns)

---

## Feature Engineering Steps

### 1. Ordinal Encoding (9 features)

Converted ordinal categorical features to numeric scales preserving natural ordering:

| Feature | Mapping | Description |
|---------|---------|-------------|
| `AgeOfVehicle_encoded` | 0 (new) → 8 (>7 years) | Vehicle age in years |
| `VehiclePrice_encoded` | 1 (<20k) → 6 (>69k) | Vehicle price bracket |
| `AgeOfPolicyHolder_encoded` | 1 (16-17) → 9 (>65) | Policy holder age bracket |
| `PastNumberOfClaims_encoded` | 0 (none) → 3 (>4) | Previous claims count |
| `Days:Policy-Accident_encoded` | 0 (none) → 4 (>30) | Days from policy start to accident |
| `Days:Policy-Claim_encoded` | 0 (none) → 3 (>30) | Days from policy start to claim |
| `NumberOfSuppliments_encoded` | 0 (none) → 3 (>5) | Number of claim supplements |
| `AddressChange-Claim_encoded` | 0 (no change) → 4 (<6mo) | Recent address change |
| `NumberOfCars_encoded` | 1 → 5 (>8) | Number of insured vehicles |

### 2. Domain-Specific Flags (11 features)

Binary indicators based on fraud domain knowledge:

**Critical Fraud Signals:**
- `no_police_report`: Police report not filed (1=Yes, 0=No)
- `no_witness`: No witnesses present
- `policy_holder_fault`: Policy holder at fault (vs third party)
- `external_agent`: External agent (vs internal)

**Temporal Patterns:**
- `quick_claim`: Claim filed within 1-7 days of policy start
- `very_quick_claim_filing`: Claim filed immediately (same day)

**Historical Patterns:**
- `has_past_claims`: Customer has claim history
- `multiple_past_claims`: 2+ previous claims

**Other Suspicious Patterns:**
- `recent_address_change`: Address changed <6 months before claim
- `many_supplements`: 3+ claim supplements filed
- `urban_accident`: Accident occurred in urban area

### 3. Interaction Features (5 features)

Combined signals that may amplify fraud risk:

| Feature | Formula | Insight |
|---------|---------|---------|
| `high_risk_combo` | `no_police + no_witness` | 0-2 scale, 2 = highest risk |
| `external_agent_holder_fault` | `external_agent × policy_holder_fault` | External agent + at-fault combination |
| `quick_claim_no_police` | `quick_claim × no_police` | Fast claim + no documentation |
| `repeat_claimer_quick` | `has_past_claims × quick_claim` | History + suspicious timing |
| `urban_no_witness` | `urban_accident × no_witness` | Urban claim without witnesses |

### 4. Temporal Features (7 features)

Time-based patterns:

- `is_weekend`: Accident on Saturday/Sunday
- `is_end_of_year`: Claim in Nov/Dec
- `claim_same_month`: Claim filed same month as accident
- `Month_numeric`: 1-12 numeric month (0 = corrupt/unknown record)
- `MonthClaimed_numeric`: 1-12 numeric month (0 = corrupt/unknown record)
- `DayOfWeek_numeric`: 0-6 (Monday=0; -1 = corrupt/unknown record)
- `DayOfWeekClaimed_numeric`: 0-6 (Monday=0; -1 = corrupt/unknown record)

### 5. Binary Encoding (6 features)

Binary categorical features:

- `PoliceReportFiled_binary`: 0=No, 1=Yes
- `WitnessPresent_binary`: 0=No, 1=Yes
- `Fault_binary`: 0=Third Party, 1=Policy Holder
- `AgentType_binary`: 0=Internal, 1=External
- `AccidentArea_binary`: 0=Rural, 1=Urban
- `Sex_binary`: 0=Female, 1=Male

### 6. One-Hot Encoding (33 features)

Nominal categorical features expanded using `drop_first=True` to avoid multicollinearity:

| Feature | Categories | Features Created |
|---------|-----------|-----------------|
| `Make_*` | 19 manufacturers | 18 features |
| `MaritalStatus_*` | 4 statuses | 3 features |
| `PolicyType_*` | 9 types | 8 features |
| `VehicleCategory_*` | 3 categories | 2 features |
| `BasePolicy_*` | 3 types | 2 features |

*Note: Used `drop_first=True` to avoid multicollinearity*

### 7. Feature Scaling

StandardScaler applied to 14 features:

**Ordinal encoded (9):** `AgeOfVehicle_encoded`, `VehiclePrice_encoded`, `AgeOfPolicyHolder_encoded`, `PastNumberOfClaims_encoded`, `Days:Policy-Accident_encoded`, `Days:Policy-Claim_encoded`, `NumberOfSuppliments_encoded`, `AddressChange-Claim_encoded`, `NumberOfCars_encoded`

**Numerical (5):** `Age`, `Deductible`, `DriverRating`, `WeekOfMonth`, `WeekOfMonthClaimed`

**Scaler saved to:** `models/scaler_v1.pkl`

---

## Final Feature Set

**Total Features:** 76 (excluding target)  
**Records:** 15,420  
**Feature Types:**

| Type | Count |
|------|-------|
| Ordinal encoded | 9 |
| Domain flags | 11 |
| Interaction features | 5 |
| Temporal features | 7 |
| Binary encoded | 6 |
| One-hot encoded | 33 |
| Numerical (scaled) | 5 |
| **Total** | **76** |

**All features are numeric** — ready for ML model training.  
**Data quality:** 0 null values, fraud rate preserved at 5.99%.

---

## Key Insights

**Most Important Engineered Features (Expected):**
1. `high_risk_combo` — Combines two strongest signals
2. `no_police_report` — Single strongest fraud indicator
3. `no_witness` — Second strongest fraud indicator
4. `quick_claim` — Suspicious timing pattern
5. `external_agent_holder_fault` — High-risk combination

**Feature Engineering Impact:**

| Stage | Features |
|-------|----------|
| Input (cleaned data) | 30 |
| After ordinal encoding | +9 encoded cols |
| After domain flags | +11 |
| After interaction features | +5 |
| After temporal features | +7 |
| After binary encoding | +6 |
| After one-hot encoding | +33 |
| After dropping originals | −24 |
| **Final (excluding target)** | **76** |

---

## Data Quality Notes

- 1 corrupt record found with `'0'` values in `DayOfWeekClaimed` and `MonthClaimed`
- Handled by mapping `'0'` → `-1` for day features and `'0'` → `0` for month features
- Record retained (not dropped) — 15,420 records preserved throughout
- Final null count: **0**

---

## Reproducibility

To reproduce feature engineering:
```bash
cd ml_pipeline
python feature_engineering.py
```

All logic in `ml_pipeline/feature_engineering.py`