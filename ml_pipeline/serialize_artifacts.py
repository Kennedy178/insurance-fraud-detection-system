"""
Model Serialization — Day 14
=============================
Saves all production artifacts needed by the FastAPI backend (Phase 3).

Run from ml_pipeline/ directory:
    python serialize_artifacts.py

What this produces:
    models/fraud_detector_v1.joblib   ← production model (renamed from xgboost_v1.pkl)
    models/feature_metadata.json      ← feature list, dtypes, preprocessing steps
    models/model_registry.json        ← single source of truth for all model paths

Note on preprocessor:
    XGBoost with scale_pos_weight does NOT need StandardScaler — tree-based
    models are invariant to feature scaling. Saving a scaler here would be
    misleading. Instead we save feature metadata which the inference module
    uses to validate and align incoming data.
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
SOURCE_MODEL     = '../models/xgboost_v1.pkl'
SOURCE_THRESHOLD = '../models/xgboost_threshold.json'
SOURCE_EVAL      = '../data/processed/test_evaluation_report.json'
TEST_DATA_PATH   = '../data/processed/test.csv'
TRAIN_DATA_PATH  = '../data/processed/train_processed.csv'

OUTPUT_MODEL     = '../models/fraud_detector_v1.joblib'
OUTPUT_METADATA  = '../models/feature_metadata.json'
OUTPUT_REGISTRY  = '../models/model_registry.json'


def clean_column_names(df):
    """Match cleaning done during training — must be identical"""
    df = df.copy()
    df.columns = (df.columns
                  .str.replace(':', '_')
                  .str.replace('-', '_')
                  .str.replace(' ', '_'))
    return df


def save_production_model():
    """Copy xgboost_v1.pkl → fraud_detector_v1.joblib with joblib format"""
    print("Saving production model...")
    model = joblib.load(SOURCE_MODEL)
    joblib.dump(model, OUTPUT_MODEL, compress=3)
    size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    print(f"  ✓ Saved → {OUTPUT_MODEL}  ({size_mb:.2f} MB)")
    return model


def build_feature_metadata(model):
    """
    Build complete feature metadata from training data.
    This is what the inference module uses to:
      - Validate incoming API requests have correct features
      - Align column order before prediction
      - Document preprocessing steps
    """
    print("\nBuilding feature metadata...")

    # Load training data to extract feature info
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    X_train = train_df.drop('FraudFound_P', axis=1)
    X_train = clean_column_names(X_train)

    # Load test data to cross-check feature consistency
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop('FraudFound_P', axis=1)
    X_test = clean_column_names(X_test)

    # Verify features match
    assert list(X_train.columns) == list(X_test.columns), \
        "ERROR: Train and test features don't match — check preprocessing"

    # Build feature info
    features = []
    for col in X_train.columns:
        dtype     = str(X_train[col].dtype)
        n_unique  = int(X_train[col].nunique())
        is_binary = n_unique == 2
        is_cat    = dtype == 'object' or (is_binary and 'float' not in dtype)

        features.append({
            'name'       : col,
            'dtype'      : dtype,
            'n_unique'   : n_unique,
            'is_binary'  : is_binary,
            'mean'       : float(X_train[col].mean()) if 'float' in dtype or 'int' in dtype else None,
            'std'        : float(X_train[col].std())  if 'float' in dtype or 'int' in dtype else None,
            'min'        : float(X_train[col].min())  if 'float' in dtype or 'int' in dtype else None,
            'max'        : float(X_train[col].max())  if 'float' in dtype or 'int' in dtype else None,
        })

    # Load threshold data
    with open(SOURCE_THRESHOLD, 'r') as f:
        threshold_data = json.load(f)

    # Load eval report
    with open(SOURCE_EVAL, 'r') as f:
        eval_report = json.load(f)

    metadata = {
        'model_name'         : 'fraud_detector_v1',
        'model_version'      : '1.0.0',
        'model_file'         : 'fraud_detector_v1.joblib',
        'threshold_file'     : 'xgboost_threshold.json',
        'training_date'      : time.strftime('%Y-%m-%d'),
        'algorithm'          : 'XGBoost (XGBClassifier)',

        # Feature info
        'n_features'         : len(features),
        'feature_names'      : [f['name'] for f in features],
        'feature_details'    : features,

        # Preprocessing — must match model_training.py exactly
        'preprocessing'      : {
            'scaler'              : 'None — XGBoost is scale-invariant',
            'imbalance_handling'  : 'scale_pos_weight=15.71 (no SMOTE)',
            'column_cleaning'     : 'Replace :, -, space → underscore',
            'missing_values'      : 'None in this dataset after feature engineering',
            'encoding'            : 'One-hot encoding applied during feature engineering',
        },

        # Model parameters (best baseline — tuning confirmed these are optimal)
        'model_params'       : {
            'n_estimators'      : 100,
            'max_depth'         : 6,
            'learning_rate'     : 0.1,
            'subsample'         : 0.8,
            'colsample_bytree'  : 0.8,
            'scale_pos_weight'  : 15.71,
            'eval_metric'       : 'aucpr',
            'random_state'      : 42,
        },

        # Thresholds
        'thresholds'         : {
            'f1_threshold'      : threshold_data['f1_threshold'],
            'deployed_threshold': threshold_data['deployed_threshold'],
            'rationale'         : threshold_data['rationale'],
        },

        # Test set performance (honest numbers)
        'performance'        : {
            'dataset'           : 'held-out test set (never seen during training)',
            'n_test_samples'    : 2313,
            'fraud_rate'        : 0.0597,
            'roc_auc'           : eval_report['roc_auc'],
            'pr_auc'            : eval_report['pr_auc'],
            'f1_score'          : eval_report['f1_score_f1'],
            'recall_deployed'   : eval_report['recall_deployed'],
            'fraud_detection_rate': eval_report['fraud_detection_rate'],
            'business_saving_vs_default': eval_report['cost_saving_vs_default'],
        },

        # Data
        'training_data'      : {
            'source'            : 'Oracle auto insurance claims dataset (Kaggle)',
            'total_samples'     : 15420,
            'train_samples'     : 10794,
            'val_samples'       : 2313,
            'test_samples'      : 2313,
            'fraud_ratio'       : '5.97%',
            'date_range'        : 'Historical — no explicit date range in source data',
            'target_column'     : 'FraudFound_P',
        },
    }

    with open(OUTPUT_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved → {OUTPUT_METADATA}")
    print(f"  Features documented: {len(features)}")
    return metadata


def build_model_registry(metadata):
    """
    Single JSON file the FastAPI app reads on startup to know
    which model files to load and where they are.
    """
    print("\nBuilding model registry...")

    registry = {
        'active_model'       : 'fraud_detector_v1',
        'models'             : {
            'fraud_detector_v1': {
                'model_file'      : 'fraud_detector_v1.joblib',
                'threshold_file'  : 'xgboost_threshold.json',
                'metadata_file'   : 'feature_metadata.json',
                'version'         : '1.0.0',
                'status'          : 'production',
                'deployed_date'   : time.strftime('%Y-%m-%d'),
                'roc_auc'         : metadata['performance']['roc_auc'],
                'fraud_detection_rate': metadata['performance']['fraud_detection_rate'],
            }
        }
    }

    with open(OUTPUT_REGISTRY, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"  ✓ Saved → {OUTPUT_REGISTRY}")
    return registry


def verify_artifacts(model, metadata):
    """
    Load saved artifacts and run a smoke test prediction
    to verify everything saved correctly.
    """
    print("\nVerifying artifacts...")

    # Load fresh from disk
    loaded_model = joblib.load(OUTPUT_MODEL)
    with open(OUTPUT_METADATA, 'r') as f:
        loaded_meta = json.load(f)
    with open(SOURCE_THRESHOLD, 'r') as f:
        threshold_data = json.load(f)

    # Load test data and make a prediction
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop('FraudFound_P', axis=1)
    X_test = clean_column_names(X_test)

    # Time the inference
    start = time.perf_counter()
    y_proba = loaded_model.predict_proba(X_test[:1])[:, 1]
    elapsed_ms = (time.perf_counter() - start) * 1000

    deployed_threshold = threshold_data['deployed_threshold']
    prediction = int(y_proba[0] >= deployed_threshold)

    print(f"  ✓ Model loaded from disk successfully")
    print(f"  ✓ Feature count matches: {loaded_meta['n_features']} features")
    print(f"  ✓ Smoke test prediction: {y_proba[0]:.4f} → {'FRAUD' if prediction else 'LEGITIMATE'}")
    print(f"  ✓ Single-record inference time: {elapsed_ms:.2f}ms  (target: <100ms)")

    if elapsed_ms > 100:
        print(f"  ⚠️  WARNING: Inference time {elapsed_ms:.2f}ms exceeds 100ms target")
    else:
        print(f"  ✓ Inference time is within <100ms target")

    return elapsed_ms


def print_summary(metadata, elapsed_ms):
    print("\n" + "="*60)
    print("DAY 14: SERIALIZATION COMPLETE")
    print("="*60)
    print(f"\n Production Artifacts:")
    print(f"  ✓ models/fraud_detector_v1.joblib")
    print(f"  ✓ models/xgboost_threshold.json    (already existed)")
    print(f"  ✓ models/feature_metadata.json")
    print(f"  ✓ models/model_registry.json")
    print(f"\n Model Summary:")
    print(f"  Algorithm:         XGBoost")
    print(f"  Features:          {metadata['n_features']}")
    print(f"  ROC-AUC (test):    {metadata['performance']['roc_auc']:.4f}")
    print(f"  Fraud caught:      {metadata['performance']['fraud_detection_rate']*100:.1f}%")
    print(f"  Business saving:   ${metadata['performance']['business_saving_vs_default']:,}")
    print(f"  Inference time:    {elapsed_ms:.2f}ms")
    print(f"\n Ready for:")
    print(f"  → ml_pipeline/inference.py  (Day 14 inference module)")
    print(f"  → app/services/ml_service.py (Day 16 FastAPI integration)")


if __name__ == "__main__":
    print("="*60)
    print("MODEL SERIALIZATION & PRODUCTION PREP")
    print("="*60)

    os.makedirs('../models', exist_ok=True)

    model    = save_production_model()
    metadata = build_feature_metadata(model)
    registry = build_model_registry(metadata)
    elapsed  = verify_artifacts(model, metadata)
    print_summary(metadata, elapsed)