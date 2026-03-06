# Model Card — Insurance Fraud Detector v1.0

**Model ID:** `fraud_detector_v1`  
**Version:** 1.0.0  
**Algorithm:** XGBoost (XGBClassifier)  
**Status:** Production  
**Last Updated:** 2025  

---

## 1. Model Overview

The Insurance Fraud Detector is an XGBoost binary classification model that
assigns a fraud probability score to automobile insurance claims. It is
designed to be integrated into the claim processing pipeline of an insurance
company, flagging high-risk claims for human investigator review before payout.

The model does not make final fraud determinations. It produces a risk score
and a binary flag that triggers a human review workflow.

---

## 2. Intended Use

### Primary Use Cases
- Real-time fraud scoring of incoming insurance claims via REST API
- Batch processing of historical claims for retrospective analysis
- Prioritisation of investigator workload (high-risk claims first)
- Dashboard analytics showing fraud trends over time

### Users
- Insurance fraud investigators (primary)
- Claims managers (dashboard analytics)
- Backend API consumers integrating fraud scoring into claim workflows

### Out-of-Scope Uses
- Final fraud determination without human review
- Use on insurance claim types other than automobile (model was trained on auto data)
- Use as the sole evidence basis for claim denial or legal action
- Deployment without threshold recalibration when business costs change

---

## 3. Training Data

| Property | Value |
|---|---|
| Source | Oracle auto insurance claims dataset (Kaggle) |
| Total samples | 15,420 claims |
| Training set | 10,794 samples (70%) |
| Validation set | 2,313 samples (15%) |
| Test set | 2,313 samples (15%) |
| Fraud rate | 5.97% (920 fraudulent claims) |
| Split method | Stratified — fraud ratio preserved in all splits |
| Date range | Historical — no explicit date range in source data |
| Target variable | `FraudFound_P` (1 = fraud, 0 = legitimate) |

### Class Imbalance Handling
SMOTE was evaluated and rejected. Synthetic oversampling introduces noise
in tabular insurance data and inflates validation metrics without improving
production performance. `scale_pos_weight = 15.71` was used instead —
this tells XGBoost to penalise missing a fraud 15.71× more than a false alarm,
without modifying the training data distribution.

---

## 4. Model Architecture & Parameters

```
Algorithm:          XGBoost (XGBClassifier)
n_estimators:       100
max_depth:          6
learning_rate:      0.1
subsample:          0.8
colsample_bytree:   0.8
scale_pos_weight:   15.71  (calculated from class distribution)
eval_metric:        aucpr  (area under precision-recall curve)
random_state:       42
```

### Hyperparameter Tuning
RandomizedSearchCV was run across 20 combinations with 5-fold stratified CV.
The tuned model showed +0.0007 F1 improvement but degraded ROC-AUC by 0.004
and caught 7 fewer frauds in production. The baseline configuration was
retained — tuning confirmed it was already at the performance ceiling for
this dataset.

---

## 5. Preprocessing Pipeline

| Step | Method | Notes |
|---|---|---|
| Missing values | None required | Dataset complete after feature engineering |
| Scaling | None | XGBoost is scale-invariant — StandardScaler not applied |
| Categorical encoding | One-hot encoding | Applied during feature engineering phase |
| Column name cleaning | Replace `:`, `-`, ` ` → `_` | Required for LightGBM compatibility |
| Class imbalance | `scale_pos_weight=15.71` | No SMOTE applied |

---

## 6. Features

**Total features:** 76

**Top 5 by SHAP importance:**

| Rank | Feature | SHAP Importance | Business Meaning |
|---|---|---|---|
| 1 | `BasePolicy_Liability` | 0.698 | Liability policy — dominant fraud signal |
| 2 | `policy_holder_fault` | 0.557 | Fault assigned to policyholder |
| 3 | `external_agent_holder_fault` | 0.530 | Third-party agent involvement |
| 4 | `Age` | 0.262 | Claimant age profile |
| 5 | `PolicyType_Sedan___Liability` | 0.242 | Sedan + liability combination |

**Key insight from SHAP:** The model detects fraud primarily through the
*absence* of patterns that legitimate claims consistently have. Negative mean
SHAP values across top features mean the model learns what normal looks like,
then flags deviations — consistent with known staged accident fraud patterns.

Full feature list: see `models/feature_metadata.json`

---

## 7. Performance — Test Set

All metrics are from the held-out test set (2,313 claims, never seen during
training, validation, or hyperparameter tuning).

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | **0.781** | Threshold-independent; baseline random = 0.50 |
| PR-AUC | **0.150** | 2.5× better than random (baseline ≈ 0.060) |
| F1-Score | **0.184** | At F1-optimised threshold (0.5586) |
| Recall (deployed) | **72.5%** | 100 of 138 frauds caught |
| Precision (deployed) | **13.8%** | Expected at 5.97% base rate |
| Brier Score | **0.128** | Probability calibration quality |

### Business Cost Analysis (Test Set)

| Strategy | Threshold | Frauds Caught | Total Cost |
|---|---|---|---|
| Default 0.50 | 0.50 | 67/138 (48.6%) | $798,800 |
| F1-Optimised | 0.5586 | 51/138 (37.0%) | $943,200 |
| **Deployed** | **0.3517** | **100/138 (72.5%)** | **$504,800** |

**Verified saving vs default: $294,000 per 2,313 claims processed**

Cost assumptions: $10,000 per missed fraud, $200 per false alarm.
Recalibrate threshold when actuary-validated costs are available.

---

## 8. Deployment Configuration

### Threshold
```json
{
  "deployed_threshold": 0.3517,
  "rationale": "Business-optimal threshold. Targets 75%+ recall. $294,000 saving vs default 0.50 on test set."
}
```

The threshold is loaded dynamically from `models/xgboost_threshold.json`.
The model cannot silently revert to 0.50 — all inference goes through
`predict_with_threshold()` or the `FraudDetector` class in `inference.py`.

### Inference
- **Model file:** `models/fraud_detector_v1.joblib`
- **Single-claim inference time:** <10ms (well within 100ms target)
- **Inference module:** `ml_pipeline/inference.py` → `FraudDetector` class
- **API integration:** `app/services/ml_service.py` (Phase 3)

---

## 9. Known Limitations

| # | Limitation | Severity | Mitigation |
|---|---|---|---|
| 1 | **Dataset size** — 15,420 claims | Medium | Retrain on production data before live deployment |
| 2 | **Imbalance ceiling** — F1 ~0.18 expected at 5.97% fraud rate | Low | Accept as structural; report ROC-AUC and business saving instead |
| 3 | **No temporal split** — fraud patterns evolve | High | Retrain on rolling 6-month window in production |
| 4 | **Assumed business costs** — $10k/$200 | High | Recalibrate threshold with actuary-validated costs |
| 5 | **Feature availability** — some features may not exist at claim submission | Medium | Build feature availability check in inference pipeline |
| 6 | **Automobile claims only** — trained on auto insurance data | High | Do not apply to health, property, or other insurance types |
| 7 | **Calibration** — slight overconfidence at high probabilities | Low | Probabilities are used as ranking scores, not literal estimates |

---

## 10. Ethical Considerations

### Fairness
The model was not audited for demographic bias. Features such as `Age` are
included, which could correlate with protected characteristics. Before live
deployment, a fairness audit should be conducted to verify the model does not
disproportionately flag claims from specific demographic groups.

### Human Oversight
This model is designed as a **decision-support tool**, not an autonomous
decision-maker. All claims flagged as HIGH risk should be reviewed by a
qualified human investigator before any adverse action is taken.

### Data Privacy
The model processes personal insurance claim data. Deployment must comply
with applicable data protection regulations (GDPR, Kenya Data Protection Act
2019, and any applicable insurance sector regulations).

### Audit Trail
All predictions must be logged with the claim ID, prediction timestamp,
risk score, threshold used, and model version. This is required for regulatory
compliance and model monitoring.

### Transparency
SHAP explanations are available for individual predictions. Investigators
should have access to the top contributing features for any flagged claim —
not just the score. This is implemented in `model_evaluation.py` and will
be integrated into the API response (Phase 3).

---

## 11. Retraining Guidelines

| Trigger | Action |
|---|---|
| Every 6 months | Full retrain on rolling window of recent claims |
| Fraud rate changes >2% | Recalculate `scale_pos_weight`, retrain |
| Business costs change | Recalibrate threshold only (no retrain needed) |
| ROC-AUC drops >0.05 on monitoring data | Immediate retrain |
| New fraud patterns detected by investigators | Feature engineering review + retrain |

---

## 12. Files & Artefacts

| File | Location | Purpose |
|---|---|---|
| Production model | `models/fraud_detector_v1.joblib` | Inference |
| Threshold config | `models/xgboost_threshold.json` | Decision boundary |
| Feature metadata | `models/feature_metadata.json` | Feature validation |
| Model registry | `models/model_registry.json` | API startup config |
| Inference module | `ml_pipeline/inference.py` | Production inference class |
| Training code | `ml_pipeline/model_training.py` | Reproducible training |
| Evaluation report | `data/processed/test_evaluation_report.json` | Test set metrics |
| SHAP importance | `data/processed/shap_feature_importance.csv` | Feature explanations |
| Evaluation notebook | `notebooks/07_model_evaluation.ipynb` | Full evaluation story |

---

*Model Card prepared as part of the Insurance Fraud Detection System — Phase 2 completion.*  
*For questions, see the evaluation notebook or contact the project maintainer.*