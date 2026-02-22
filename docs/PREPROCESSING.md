# Data Preprocessing Documentation

**Date:** February 22, 2026  
**Input:** `data/raw/insurance_claims_raw.csv`  
**Output:** `data/processed/insurance_claims_cleaned.csv`

---

## Preprocessing Steps Applied

### 1. Duplicate Removal
- **Action:** Removed exact duplicate rows
- **Records removed:** 0
- **Reason:** Avoid training on duplicate data

### 2. Missing Value Handling

#### Age Column
- **Issue:** 320 records with Age = 0 (invalid)
- **Solution:** Replaced with median age (39.0 years)
- **Reason:** Age = 0 is data entry error, not missing

#### Categorical "none" Values
- **Issue:** String "none" present in: `Days:Policy-Accident`, `Days:Policy-Claim`, `PastNumberOfClaims`, `NumberOfSuppliments`
- **Solution:** Kept as valid category
- **Reason:** Represents "no past claims" or "no time gap" — meaningful information, not true missing data

### 3. Outlier Handling
- **Method:** Capped at IQR-based bounds (1.5×IQR)
- **Columns affected:** Age, Deductible

| Column | Low Outliers Capped | High Outliers Capped | Capped Range |
|--------|--------------------:|---------------------:|-------------|
| Age | 123 (below 21.0) | 116 (above 76.0) | [21.0, 76.0] |
| Deductible | 8 (below 400.0) | 0 | [400.0, 700.0] |

- **Reason:** Extreme outliers can skew model training, but values still carry information so capping is preferred over removal

### 4. Feature Removal

| Feature | Reason for Removal |
|---------|-------------------|
| PolicyNumber | Unique ID — no predictive power |
| Year | No meaningful variation (only 1994–1996) |
| RepNumber | Representative assignment — not predictive of fraud |
| FraudFound | Dropped original string column; binary version `FraudFound_P` retained as target |

- **Features removed:** 4  
- **Remaining features after removal:** 30

### 5. Data Type Validation
- **Numeric columns:** 6
- **Categorical columns:** 24
- **Reason:** Ensure correct data types for downstream encoding and modeling

---

## Final Dataset Summary

| Property | Value |
|----------|-------|
| Initial Shape | 15,420 rows × 34 columns |
| Final Shape | 15,420 rows × 30 columns |
| Records Removed | 0 |
| Features Removed | 4 |
| Fraud Cases | 923 |
| Legitimate Cases | 14,497 |
| Fraud Rate | 5.99% |
| Null Values | None |
| Outliers | Capped (not removed) |

**Final columns (30):**  
`Month`, `WeekOfMonth`, `DayOfWeek`, `Make`, `AccidentArea`, `DayOfWeekClaimed`, `MonthClaimed`, `WeekOfMonthClaimed`, `Sex`, `MaritalStatus`, `Age`, `Fault`, `PolicyType`, `VehicleCategory`, `VehiclePrice`, `Deductible`, `DriverRating`, `Days:Policy-Accident`, `Days:Policy-Claim`, `PastNumberOfClaims`, `AgeOfVehicle`, `AgeOfPolicyHolder`, `PoliceReportFiled`, `WitnessPresent`, `AgentType`, `NumberOfSuppliments`, `AddressChange-Claim`, `NumberOfCars`, `BasePolicy`, `FraudFound_P`

**Ready for:** Feature engineering (Day 6–7)

---

## Reproducibility

To reproduce preprocessing:
```bash
cd ml_pipeline
python preprocessor.py
```

All preprocessing logic is in `ml_pipeline/preprocessor.py`