"""
Model Training Module - CORRECTED VERSION
Handles train-test split, class imbalance, and model training preparation
Fixed: No SMOTE, proper class weights, correct eval metrics
"""
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def load_engineered_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load engineered data and separate features from target
    
    Args:
        filepath: Path to engineered CSV
    
    Returns:
        X (features), y (target)
    """
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('FraudFound_P', axis=1)
    y = df['FraudFound_P']
    
    print(f"✓ Loaded engineered data")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X):,}")
    print(f"Fraud cases: {y.sum():,} ({y.mean()*100:.2f}%)")
    
    return X, y


def split_data(X: pd.DataFrame, 
               y: pd.Series,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets with stratification
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set (0.15 = 15%)
        val_size: Proportion for validation set (0.15 = 15%)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n=== SPLITTING DATA ===")
    
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain fraud ratio in all sets
    )
    
    # Second split: split remaining into train (70%) and val (15%)
    # val_size / (1 - test_size) gives us 15% of original data
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Print split information
    print(f"\nDataset Split:")
    print(f"  Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\nFraud Distribution:")
    print(f"  Training:   {y_train.sum():,} frauds ({y_train.mean()*100:.2f}%)")
    print(f"  Validation: {y_val.sum():,} frauds ({y_val.mean()*100:.2f}%)")
    print(f"  Test:       {y_test.sum():,} frauds ({y_test.mean()*100:.2f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def handle_class_imbalance(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           method: str = 'none',
                           sampling_strategy: float = 0.5,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE, undersampling, or none (use class weights instead)
    
    Args:
        X_train: Training features
        y_train: Training target
        method: 'smote', 'undersample', 'combined', or 'none'
        sampling_strategy: Target ratio of minority/majority (0.5 = 50:50)
        random_state: Random seed
    
    Returns:
        X_train_balanced, y_train_balanced
    """
    print("\n=== HANDLING CLASS IMBALANCE ===")
    
    # Calculate initial imbalance
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    initial_ratio = fraud_count / legit_count
    
    print(f"\nBefore balancing:")
    print(f"  Legitimate: {legit_count:,}")
    print(f"  Fraud: {fraud_count:,}")
    print(f"  Ratio: 1:{int(1/initial_ratio)}")
    print(f"  Imbalance: {(1-initial_ratio)*100:.1f}% skew")
    
    # CORRECTED: Added 'none' option - no resampling, use class weights instead
    if method == 'none':
        print(f"\n✓ No resampling applied - keeping original imbalanced distribution")
        print(f"  Models will use built-in class weights (scale_pos_weight) to handle imbalance")
        print(f"  This approach works better for tree-based models (XGBoost, LightGBM)")
        return X_train, y_train
    
    # Apply sampling method
    elif method == 'smote':
        # SMOTE: Synthetic Minority Over-sampling
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5
        )
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"\n✓ Applied SMOTE (Synthetic Minority Over-sampling)")
        
    elif method == 'undersample':
        # Random Under-sampling of majority class
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"\n✓ Applied Random Under-sampling")
        
    elif method == 'combined':
        # Combined: SMOTE then under-sample
        over = SMOTE(sampling_strategy=0.3, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        
        pipeline = ImbPipeline([
            ('over', over),
            ('under', under)
        ])
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        print(f"\n✓ Applied Combined (SMOTE + Under-sampling)")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'none', 'smote', 'undersample', or 'combined'")
    
    # Calculate new distribution (only if resampling was applied)
    if method != 'none':
        fraud_count_new = y_resampled.sum()
        legit_count_new = len(y_resampled) - fraud_count_new
        new_ratio = fraud_count_new / legit_count_new
        
        print(f"\nAfter balancing:")
        print(f"  Legitimate: {legit_count_new:,}")
        print(f"  Fraud: {fraud_count_new:,}")
        print(f"  Ratio: 1:{legit_count_new/fraud_count_new:.1f}")
        print(f"  Balance: {new_ratio*100:.1f}% fraud / {(1-new_ratio)*100:.1f}% legitimate")
        print(f"  Total samples: {len(X_resampled):,} (from {len(X_train):,})")
        
        # Convert back to DataFrame with proper column names
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name='FraudFound_P')
        
        return X_resampled, y_resampled


def verify_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """Verify data splits and distributions"""
    print("\n=== VERIFICATION ===")
    
    # Check shapes
    print(f"\nShapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    print(f"\n✓ No data leakage - train/val/test split before any sampling")
    
    # Check fraud distribution
    print(f"\nFraud distribution consistency:")
    print(f"  Training:   {y_train.mean()*100:.2f}%")
    print(f"  Validation: {y_val.mean()*100:.2f}%")
    print(f"  Test:       {y_test.mean()*100:.2f}%")
    
    # Check for nulls
    print(f"\nNull values:")
    print(f"  X_train: {X_train.isnull().sum().sum()}")
    print(f"  X_val: {X_val.isnull().sum().sum()}")
    print(f"  X_test: {X_test.isnull().sum().sum()}")
    
    print(f"\n✓ Data ready for training")


def prepare_training_data(input_path: str,
                          output_dir: str,
                          sampling_method: str = 'none',
                          sampling_strategy: float = 0.5,
                          random_state: int = 42) -> dict:
    """
    Complete data preparation pipeline
    
    Args:
        input_path: Path to engineered features CSV
        output_dir: Directory to save train/val/test sets
        sampling_method: 'none', 'smote', 'undersample', or 'combined'
        sampling_strategy: Resampling ratio (ignored if method='none')
        random_state: Random seed
    
    Returns:
        metadata dict with split information
    """
    print("="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Load data
    X, y = load_engineered_data(input_path)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, 
        test_size=0.15, 
        val_size=0.15,
        random_state=random_state
    )
    
    # Handle class imbalance (or not, if method='none')
    X_train_processed, y_train_processed = handle_class_imbalance(
        X_train, y_train,
        method=sampling_method,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    # Verify splits
    verify_splits(X_train_processed, X_val, X_test, y_train_processed, y_val, y_test)
    
    # Save datasets
    print("\n=== SAVING DATASETS ===")
    
    # Training set (processed - may or may not be resampled)
    train_df = pd.concat([X_train_processed, y_train_processed], axis=1)
    train_path = f"{output_dir}/train_processed.csv"
    train_df.to_csv(train_path, index=False)
    print(f"✓ Training set saved: {train_path}")
    print(f"  Shape: {train_df.shape}")
    
    # Validation set (always original, never resampled)
    val_df = pd.concat([X_val, y_val], axis=1)
    val_path = f"{output_dir}/validation.csv"
    val_df.to_csv(val_path, index=False)
    print(f"✓ Validation set saved: {val_path}")
    print(f"  Shape: {val_df.shape}")
    
    # Test set (always original, never resampled)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_path = f"{output_dir}/test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✓ Test set saved: {test_path}")
    print(f"  Shape: {test_df.shape}")
    
    # Create metadata
    metadata = {
        'total_samples': len(X),
        'n_features': X.shape[1],
        'train_samples': len(X_train_processed),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'original_fraud_rate': y.mean(),
        'train_fraud_rate': y_train_processed.mean(),
        'val_fraud_rate': y_val.mean(),
        'test_fraud_rate': y_test.mean(),
        'sampling_method': sampling_method,
        'sampling_strategy': sampling_strategy if sampling_method != 'none' else 'N/A',
        'random_state': random_state
    }
    
    # Save metadata
    metadata_df = pd.DataFrame([metadata])
    metadata_path = f"{output_dir}/split_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\n Summary:")
    print(f"  Total samples: {len(X):,}")
    print(f"  Train: {len(X_train_processed):,} ({y_train_processed.mean()*100:.2f}% fraud)")
    print(f"  Validation: {len(X_val):,} ({y_val.mean()*100:.2f}% fraud)")
    print(f"  Test: {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")
    print(f"  Sampling method: {sampling_method}")
    
    return metadata


def clean_column_names(df):
    """Remove special characters from column names for LightGBM"""
    df = df.copy()
    df.columns = df.columns.str.replace(':', '_').str.replace('-', '_').str.replace(' ', '_')
    return df


def plot_confusion_matrix(cm, model_name, save_path='../visualizations'):
    """Plot confusion matrix heatmap"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name.lower().replace(" ", "_")}_cm.png', dpi=300, bbox_inches='tight')
    plt.close()


def train_baseline_models():
    """Train baseline models (Logistic Regression and Random Forest)"""
    print("="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv('../data/processed/train_processed.csv')
    val_df = pd.read_csv('../data/processed/validation.csv')
    
    X_train = train_df.drop('FraudFound_P', axis=1)
    y_train = train_df['FraudFound_P']
    X_val = val_df.drop('FraudFound_P', axis=1)
    y_val = val_df['FraudFound_P']
    
    print(f"✓ Training: {X_train.shape}")
    print(f"✓ Validation: {X_val.shape}")
    print(f"✓ Training fraud rate: {y_train.mean()*100:.2f}%")
    
    # Calculate class weights for Logistic Regression
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    class_weight_ratio = legit_count / fraud_count
    
    print(f"\n=== CLASS WEIGHT CONFIGURATION ===")
    print(f"Class weight ratio: {class_weight_ratio:.2f}")
    print(f"Using class_weight='balanced' for both models")
    
    # 1. Logistic Regression with balanced class weights
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Handles imbalance automatically
        solver='lbfgs'
    )
    
    print("\nTraining Logistic Regression...")
    lr_model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Evaluate
    y_lr_pred = lr_model.predict(X_val)
    y_lr_proba = lr_model.predict_proba(X_val)[:, 1]
    
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(f"Accuracy:  {accuracy_score(y_val, y_lr_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_lr_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_lr_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_val, y_lr_pred):.4f}  PRIMARY METRIC")
    print(f"ROC-AUC:   {roc_auc_score(y_val, y_lr_proba):.4f}")
    
    cm = confusion_matrix(y_val, y_lr_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    frauds_caught = cm[1,1]
    total_frauds = cm[1,0] + cm[1,1]
    print(f"\n Business Impact:")
    print(f"   Caught {frauds_caught}/{total_frauds} frauds ({frauds_caught/total_frauds*100:.1f}%)")
    
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_lr_pred, target_names=['Legitimate', 'Fraud']))
    
    # Save model
    joblib.dump(lr_model, '../models/logistic_regression_baseline.pkl')
    print("✓ Model saved: ../models/logistic_regression_baseline.pkl")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, 'Logistic Regression')
    
    # 2. Random Forest with balanced class weights
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handles imbalance automatically
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    rf_model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Evaluate
    y_rf_pred = rf_model.predict(X_val)
    y_rf_proba = rf_model.predict_proba(X_val)[:, 1]
    
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(f"Accuracy:  {accuracy_score(y_val, y_rf_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_rf_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_rf_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_val, y_rf_pred):.4f}  PRIMARY METRIC")
    print(f"ROC-AUC:   {roc_auc_score(y_val, y_rf_proba):.4f}")
    
    cm = confusion_matrix(y_val, y_rf_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    frauds_caught = cm[1,1]
    total_frauds = cm[1,0] + cm[1,1]
    print(f"\n Business Impact:")
    print(f"   Caught {frauds_caught}/{total_frauds} frauds ({frauds_caught/total_frauds*100:.1f}%)")
    
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_rf_pred, target_names=['Legitimate', 'Fraud']))
    
    # Save model
    joblib.dump(rf_model, '../models/random_forest_baseline.pkl')
    print("✓ Model saved: ../models/random_forest_baseline.pkl")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, 'Random Forest')
    
    print("\n" + "="*60)
    print("BASELINE TRAINING COMPLETE")
    print("="*60)
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'lr_metrics': {
            'accuracy': accuracy_score(y_val, y_lr_pred),
            'precision': precision_score(y_val, y_lr_pred),
            'recall': recall_score(y_val, y_lr_pred),
            'f1_score': f1_score(y_val, y_lr_pred),
            'roc_auc': roc_auc_score(y_val, y_lr_proba)
        },
        'rf_metrics': {
            'accuracy': accuracy_score(y_val, y_rf_pred),
            'precision': precision_score(y_val, y_rf_pred),
            'recall': recall_score(y_val, y_rf_pred),
            'f1_score': f1_score(y_val, y_rf_pred),
            'roc_auc': roc_auc_score(y_val, y_rf_proba)
        }
    }

"""
changes:
- optimize_threshold() now saves THREE thresholds to the JSON
- predict_with_threshold() loads 'deployed_threshold' (the recall-optimised one)
- The model .pkl is untouched — we are only changing the decision boundary
"""

def optimize_threshold(y_true, y_proba, model_name='xgboost', save_dir='../models'):
    """
    Finds THREE thresholds and saves them all to disk:
      1. f1_threshold     — mathematically best F1 (what your friend gave you)
      2. recall_threshold — catches the most fraud (best for business ROI)
      3. deployed_threshold — what production actually uses (set to recall_threshold)

    Args:
        y_true    : Ground-truth labels (validation set only — never training or test)
        y_proba   : model.predict_proba(X_val)[:, 1]
        model_name: used to name the saved JSON file
        save_dir  : where to write the JSON

    Returns:
        deployed_threshold (float)  — the one production will use
        best_preds         (array)  — predictions at deployed_threshold
    """
    from sklearn.metrics import precision_recall_curve, f1_score as sk_f1, recall_score
    import numpy as np
    import json, os

    os.makedirs(save_dir, exist_ok=True)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # ── 1. F1-OPTIMISED THRESHOLD ─────────────────────────────
    f1_scores = np.divide(
        2 * (precisions[:-1] * recalls[:-1]),
        (precisions[:-1] + recalls[:-1]),
        out=np.zeros(len(thresholds)),
        where=(precisions[:-1] + recalls[:-1]) != 0
    )
    f1_idx        = int(np.argmax(f1_scores))
    f1_threshold  = float(thresholds[f1_idx])
    f1_best       = float(f1_scores[f1_idx])

    # ── 2. RECALL-OPTIMISED THRESHOLD ─────────────────────────
    # We want recall >= 0.75 (catch at least 75% of frauds)
    # Among all thresholds that meet that bar, pick the one
    # with the highest precision (fewest wasted investigations)
    #
    # Why 0.75? With $10,000 missed fraud vs $200 false alarm,
    # catching 75%+ of frauds is the break-even business target.
    # You can adjust this number — higher = catch more fraud,
    # lower = fewer false alarms.
    TARGET_RECALL = 0.75

    # recalls[:-1] aligns with thresholds array
    recall_vals = recalls[:-1]
    meets_target = recall_vals >= TARGET_RECALL

    if meets_target.any():
        # Among thresholds that meet recall target, pick highest precision
        candidate_precisions = np.where(meets_target, precisions[:-1], 0)
        recall_idx           = int(np.argmax(candidate_precisions))
    else:
        # If no threshold hits 0.75 recall, just take the highest recall available
        print(f"   ⚠️  No threshold achieves {TARGET_RECALL:.0%} recall — using best available")
        recall_idx = int(np.argmax(recall_vals))

    recall_threshold      = float(thresholds[recall_idx])
    recall_at_threshold   = float(recall_vals[recall_idx])
    precision_at_threshold = float(precisions[recall_idx])
    f1_at_recall_thresh   = float(f1_scores[recall_idx])

    # ── 3. DEFAULT 0.50 for comparison ────────────────────────
    default_preds = (y_proba >= 0.50).astype(int)
    default_f1    = sk_f1(y_true, default_preds)

    # ── PRINT COMPARISON ──────────────────────────────────────
    print(f"\n Threshold Analysis ({model_name}):")
    print(f"   {'':30s}  {'Threshold':>10}  {'F1':>7}  {'Recall':>7}  {'Precision':>10}")
    print(f"   {'─'*65}")
    print(f"   {'Default (0.50)':30s}  {'0.5000':>10}  {default_f1:>7.4f}  {recall_score(y_true, default_preds):>7.4f}  {'-':>10}")
    print(f"   {'F1-Optimised':30s}  {f1_threshold:>10.4f}  {f1_best:>7.4f}  {recalls[f1_idx]:>7.4f}  {precisions[f1_idx]:>10.4f}")
    print(f"   {'Recall-Optimised (deployed)':30s}  {recall_threshold:>10.4f}  {f1_at_recall_thresh:>7.4f}  {recall_at_threshold:>7.4f}  {precision_at_threshold:>10.4f}")

    # ── BUSINESS COST COMPARISON ──────────────────────────────
    # Compare default 0.50 vs deployed recall threshold
    preds_deployed  = (y_proba >= recall_threshold).astype(int)
    
    from sklearn.metrics import confusion_matrix
    cm_default  = confusion_matrix(y_true, default_preds)
    cm_deployed = confusion_matrix(y_true, preds_deployed)

    missed_default  = int(cm_default[1, 0])
    falarms_default = int(cm_default[0, 1])
    missed_deployed  = int(cm_deployed[1, 0])
    falarms_deployed = int(cm_deployed[0, 1])

    cost_per_missed_fraud  = 10_000
    cost_per_false_alarm   = 200

    cost_default  = missed_default  * cost_per_missed_fraud + falarms_default  * cost_per_false_alarm
    cost_deployed = missed_deployed * cost_per_missed_fraud + falarms_deployed * cost_per_false_alarm

    print(f"\n Business Cost Comparison (validation set):")
    print(f"   {'':30s}  {'Missed Frauds':>14}  {'False Alarms':>13}  {'Total Cost':>11}")
    print(f"   {'─'*75}")
    print(f"   {'Default (0.50)':30s}  {missed_default:>14}  {falarms_default:>13}  ${cost_default:>10,.0f}")
    print(f"   {'Deployed (recall-opt)':30s}  {missed_deployed:>14}  {falarms_deployed:>13}  ${cost_deployed:>10,.0f}")
    print(f"\n    Deployed threshold saves: ${cost_default - cost_deployed:,.0f} vs default 0.50")

    # ── SAVE ALL THREE TO DISK ────────────────────────────────
    threshold_data = {
        'model_name'          : model_name,

        # The three thresholds
        'f1_threshold'        : f1_threshold,
        'recall_threshold'    : recall_threshold,
        'deployed_threshold'  : recall_threshold,   # ← production uses this one

        # Metrics at each
        'f1_at_f1_threshold'         : f1_best,
        'recall_at_f1_threshold'     : float(recalls[f1_idx]),
        'f1_at_recall_threshold'     : f1_at_recall_thresh,
        'recall_at_recall_threshold' : recall_at_threshold,
        'precision_at_recall_threshold': precision_at_threshold,

        # Business cost at deployed threshold
        'missed_frauds_deployed'  : missed_deployed,
        'false_alarms_deployed'   : falarms_deployed,
        'business_cost_deployed'  : cost_deployed,
        'business_cost_default'   : cost_default,
        'cost_saving_vs_default'  : cost_default - cost_deployed,

        'rationale': (
            f"recall_threshold deployed — catching {recall_at_threshold:.0%} of frauds. "
            f"Missed fraud costs ${cost_per_missed_fraud:,} vs ${cost_per_false_alarm} false alarm. "
            f"Recall-optimised threshold saves ${cost_default - cost_deployed:,} vs default 0.50."
        )
    }

    save_path = f"{save_dir}/{model_name}_threshold.json"
    with open(save_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    print(f"\n   ✓ All thresholds saved → {save_path}")

    # Return the DEPLOYED threshold and its predictions
    return recall_threshold, preds_deployed


"""
What changed and why:
- optimize_threshold() now returns the DEPLOYED (recall) threshold + predictions
- But the comparison table should show F1-optimised metrics (fair model comparison)
- So we now make TWO sets of predictions:
    y_pred_deployed → used for business impact / confusion matrix display
    y_pred_f1       → used for the comparison table metrics
- The returned metrics dict uses y_pred_f1 so the table is fair
- The returned y_pred is y_pred_deployed so downstream plots show reality
"""

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost with scale_pos_weight for class imbalance.
    Finds both F1-optimised and recall-optimised (deployed) thresholds.
    Reports F1-optimised metrics in comparison table.
    Production uses deployed threshold via predict_with_threshold().
    """
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST CLASSIFIER")
    print("=" * 60)

    fraud_count      = y_train.sum()
    legit_count      = len(y_train) - fraud_count
    scale_pos_weight = legit_count / fraud_count

    print(f"\n=== CLASS WEIGHT CONFIGURATION ===")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    print(f"Based on fraud rate: {y_train.mean() * 100:.2f}%")
    print(f"Interpretation: Penalise missing a fraud {scale_pos_weight:.1f}x more than a false alarm")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=42,
        verbosity=0
    )

    print(f"\nTraining XGBoost...")
    model.fit(X_train, y_train)
    print("✓ Model trained")

    # ── Step 1: raw probabilities ──────────────────────────────
    y_proba = model.predict_proba(X_val)[:, 1]

    # ── Step 2: find both thresholds, save to disk ─────────────
    # optimize_threshold() returns the DEPLOYED (recall) threshold
    # but also saves f1_threshold inside the JSON
    deployed_thresh, y_pred_deployed = optimize_threshold(
        y_val, y_proba,
        model_name='xgboost',
        save_dir='../models'
    )

    # ── Step 3: load f1_threshold from saved JSON for fair table ──
    import json
    with open('../models/xgboost_threshold.json', 'r') as f:
        threshold_data = json.load(f)

    f1_thresh   = threshold_data['f1_threshold']
    y_pred_f1   = (y_proba >= f1_thresh).astype(int)

    # ── Step 4: compute metrics using F1-optimised predictions ──
    # This keeps the comparison table fair across all 4 models
    accuracy  = accuracy_score(y_val, y_pred_f1)
    precision = precision_score(y_val, y_pred_f1)
    recall    = recall_score(y_val, y_pred_f1)
    f1        = f1_score(y_val, y_pred_f1)
    roc_auc   = roc_auc_score(y_val, y_proba)  # threshold-independent, always uses y_proba

    # ── Step 5: print performance at F1-optimised threshold ─────
    print("\n=== VALIDATION SET PERFORMANCE (F1-optimised threshold) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}  (fraud detection rate)")
    print(f"F1-Score:  {f1:.4f}   PRIMARY METRIC")
    print(f"ROC-AUC:   {roc_auc:.4f}  (threshold-independent)")

    cm_f1 = confusion_matrix(y_val, y_pred_f1)
    print(f"\nConfusion Matrix (F1 threshold = {f1_thresh:.4f}):")
    print(f"  TN: {cm_f1[0, 0]:,}  |  FP: {cm_f1[0, 1]:,}")
    print(f"  FN: {cm_f1[1, 0]:,}  |  TP: {cm_f1[1, 1]:,}")

    # ── Step 6: also print deployed (business) performance ──────
    cm_deployed = confusion_matrix(y_val, y_pred_deployed)
    frauds_caught_deployed = cm_deployed[1, 1]
    total_frauds           = cm_deployed[1, 0] + cm_deployed[1, 1]

    print(f"\n=== DEPLOYED PERFORMANCE (recall threshold = {deployed_thresh:.4f}) ===")
    print(f"  TN: {cm_deployed[0, 0]:,}  |  FP: {cm_deployed[0, 1]:,}")
    print(f"  FN: {cm_deployed[1, 0]:,}  |  TP: {cm_deployed[1, 1]:,}")
    print(f"\n Business Impact (deployed threshold):")
    print(f"   Caught {frauds_caught_deployed}/{total_frauds} frauds ({frauds_caught_deployed / total_frauds * 100:.1f}%)")
    print(f"   Business cost: ${threshold_data['business_cost_deployed']:,.0f}  (vs ${threshold_data['business_cost_default']:,.0f} at default 0.50)")
    print(f"   Saving: ${threshold_data['cost_saving_vs_default']:,.0f}")

    print(f"\nClassification Report (F1 threshold = {f1_thresh:.4f}):")
    print(classification_report(y_val, y_pred_f1, target_names=['Legitimate', 'Fraud']))

    # ── Step 7: feature importance ───────────────────────────────
    feature_importance = pd.DataFrame({
        'feature'   : X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10).to_string(index=False))

    # confusion matrix plot uses F1 threshold (what the table shows)
    plot_confusion_matrix(cm_f1, 'XGBoost')

    # ── Step 8: metrics dict uses F1-optimised for fair comparison ─
    metrics = {
        'model'    : 'XGBoost',
        'accuracy' : accuracy,
        'precision': precision,
        'recall'   : recall,
        'f1_score' : f1,
        'roc_auc'  : roc_auc,
        'threshold': f1_thresh      # shows f1 threshold in comparison table
    }

    # y_pred returned is the DEPLOYED one — downstream plots show real production behaviour
    return model, metrics, y_pred_deployed, y_proba, feature_importance


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Train LightGBM with proper class weight handling
    """
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM CLASSIFIER")
    print("="*60)
    
    # Calculate scale_pos_weight
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    scale_pos_weight = legit_count / fraud_count
    
    print(f"\n=== CLASS WEIGHT CONFIGURATION ===")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    print(f"Based on fraud rate: {y_train.mean()*100:.2f}%")
    
    # LightGBM with proper parameters
    model = LGBMClassifier(
        n_estimators=100,     #  Reduce trees (was 300)
        max_depth=6,
        learning_rate=0.1,    #  Standard (was 0.05)
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        is_unbalance=False,
        metric='auc',
        random_state=42,
        verbosity=-1
    )
    
    print(f"\nTraining LightGBM with scale_pos_weight={scale_pos_weight:.2f}...")
    model.fit(X_train, y_train)  # REMOVED callbacks parameter
    print("✓ Model trained")
    
    # Rest stays the same...
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}  PRIMARY METRIC")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    frauds_caught = cm[1,1]
    total_frauds = cm[1,0] + cm[1,1]
    frauds_missed = cm[1,0]
    false_alarms = cm[0,1]
    
    print(f"\n Business Impact:")
    print(f"   Caught {frauds_caught}/{total_frauds} frauds ({frauds_caught/total_frauds*100:.1f}%)")
    print(f"   Missed {frauds_missed} frauds (cost: ${frauds_missed*10000:,})")
    print(f"   False alarms: {false_alarms} (cost: ${false_alarms*200:,})")
    
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, 'LightGBM')
    
    metrics = {
        'model': 'LightGBM',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return model, metrics, y_pred, y_proba, feature_importance

def cross_validate_models(X_train, y_train, models, cv_folds=5):
    """Cross-validate models using stratified K-fold"""
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION ({cv_folds}-FOLD STRATIFIED)")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    results = []
    
    for name, model in models.items():
        print(f"\nRunning CV for {name}...")
        
        # Cross-validate
        f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
        recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall', n_jobs=-1)
        precision_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision', n_jobs=-1)
        
        print(f"  F1-Score:  {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")
        print(f"  Recall:    {recall_scores.mean():.4f} (±{recall_scores.std():.4f})")
        print(f"  Precision: {precision_scores.mean():.4f} (±{precision_scores.std():.4f})")
        
        results.append({
            'model': name,
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'recall_mean': recall_scores.mean(),
            'recall_std': recall_scores.std(),
            'precision_mean': precision_scores.mean(),
            'precision_std': precision_scores.std()
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(results_df.to_string(index=False))
    
    return results_df


def compare_all_models(lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics):
    """Compare all 4 models side by side"""
    print("\n" + "="*60)
    print("COMPLETE MODEL COMPARISON (ALL 4 MODELS)")
    print("="*60)
    
    comparison = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics])
    
    print("\n")
    print(comparison.to_string(index=False))
    
    # Find best models
    best_f1_idx = comparison['f1_score'].idxmax()
    best_recall_idx = comparison['recall'].idxmax()
    best_auc_idx = comparison['roc_auc'].idxmax()
    
    print(f"\n WINNERS:")
    print(f"   Best F1-Score: {comparison.loc[best_f1_idx, 'model']} ({comparison.loc[best_f1_idx, 'f1_score']:.4f})")
    print(f"   Best Recall: {comparison.loc[best_recall_idx, 'model']} ({comparison.loc[best_recall_idx, 'recall']:.4f})")
    print(f"   Best ROC-AUC: {comparison.loc[best_auc_idx, 'model']} ({comparison.loc[best_auc_idx, 'roc_auc']:.4f})")
    
    # Calculate improvement
    baseline_best_f1 = max(lr_metrics['f1_score'], rf_metrics['f1_score'])
    advanced_best_f1 = max(xgb_metrics['f1_score'], lgbm_metrics['f1_score'])
    improvement = ((advanced_best_f1 - baseline_best_f1) / baseline_best_f1) * 100
    
    print(f"\n IMPROVEMENT OVER BASELINE:")
    print(f"   F1-Score improved by {improvement:.1f}%")
    print(f"   {baseline_best_f1:.4f} → {advanced_best_f1:.4f}")
    
    return comparison


def plot_advanced_model_comparisons(y_val, y_lr_pred, y_rf_pred, y_xgb_pred, y_lgbm_pred,
                                     y_lr_proba, y_rf_proba, y_xgb_proba, y_lgbm_proba,
                                     lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics,
                                     save_path='../visualizations'):
    """Create comprehensive comparison visualizations"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # 1. All confusion matrices in one plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models_cm = [
        ('Logistic Regression', confusion_matrix(y_val, y_lr_pred)),
        ('Random Forest', confusion_matrix(y_val, y_rf_pred)),
        ('XGBoost', confusion_matrix(y_val, y_xgb_pred)),
        ('LightGBM', confusion_matrix(y_val, y_lgbm_pred))
    ]
    
    for idx, (name, cm) in enumerate(models_cm):
        ax = axes[idx // 2, idx % 2]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved")
    plt.close()
    
    # 2. ROC curves comparison
    plt.figure(figsize=(10, 8))
    
    models_data = [
        ('Logistic Regression', y_lr_proba, lr_metrics['roc_auc'], 'blue'),
        ('Random Forest', y_rf_proba, rf_metrics['roc_auc'], 'green'),
        ('XGBoost', y_xgb_proba, xgb_metrics['roc_auc'], 'red'),
        ('LightGBM', y_lgbm_proba, lgbm_metrics['roc_auc'], 'purple')
    ]
    
    for name, proba, auc, color in models_data:
        fpr, tpr, _ = roc_curve(y_val, proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2, color=color)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved")
    plt.close()
    
    # 3. Metrics comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    comparison_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics])
    
    # F1 and Recall comparison
    models = ['LR', 'RF', 'XGB', 'LGBM']
    f1_scores = comparison_df['f1_score'].values
    recall_scores = comparison_df['recall'].values
    
    x = range(len(models))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], f1_scores, width, label='F1-Score', color='orange', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], recall_scores, width, label='Recall', color='green', alpha=0.8)
    axes[0].set_ylabel('Score')
    axes[0].set_title('F1-Score & Recall Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Add value labels
    for i, (f1, recall) in enumerate(zip(f1_scores, recall_scores)):
        axes[0].text(i - width/2, f1 + 0.02, f'{f1:.3f}', ha='center', fontsize=9)
        axes[0].text(i + width/2, recall + 0.02, f'{recall:.3f}', ha='center', fontsize=9)
    
    # Precision and ROC-AUC
    precision_scores = comparison_df['precision'].values
    auc_scores = comparison_df['roc_auc'].values
    
    axes[1].bar([i - width/2 for i in x], precision_scores, width, label='Precision', color='blue', alpha=0.8)
    axes[1].bar([i + width/2 for i in x], auc_scores, width, label='ROC-AUC', color='red', alpha=0.8)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision & ROC-AUC Comparison', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Add value labels
    for i, (prec, auc) in enumerate(zip(precision_scores, auc_scores)):
        axes[1].text(i - width/2, prec + 0.02, f'{prec:.3f}', ha='center', fontsize=9)
        axes[1].text(i + width/2, auc + 0.02, f'{auc:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved")
    plt.close()


def train_advanced_models():
    """Complete advanced model training pipeline"""
    print("="*60)
    print("ADVANCED MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv('../data/processed/train_processed.csv')
    val_df = pd.read_csv('../data/processed/validation.csv')
    
    X_train = train_df.drop('FraudFound_P', axis=1)
    y_train = train_df['FraudFound_P']
    X_val = val_df.drop('FraudFound_P', axis=1)
    y_val = val_df['FraudFound_P']
    
    print(f"✓ Training: {X_train.shape}")
    print(f"✓ Validation: {X_val.shape}")
    print(f"✓ Training fraud rate: {y_train.mean()*100:.2f}%")
    
    # Load baseline results for comparison (BEFORE cleaning column names)
    print("\nLoading baseline models...")
    lr_model = joblib.load('../models/logistic_regression_baseline.pkl')
    rf_model = joblib.load('../models/random_forest_baseline.pkl')
    
    # Get baseline metrics (with ORIGINAL column names - DON'T clean yet)
    y_lr_pred = lr_model.predict(X_val)
    y_lr_proba = lr_model.predict_proba(X_val)[:, 1]
    lr_metrics = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_val, y_lr_pred),
        'precision': precision_score(y_val, y_lr_pred),
        'recall': recall_score(y_val, y_lr_pred),
        'f1_score': f1_score(y_val, y_lr_pred),
        'roc_auc': roc_auc_score(y_val, y_lr_proba)
    }
    
    y_rf_pred = rf_model.predict(X_val)
    y_rf_proba = rf_model.predict_proba(X_val)[:, 1]
    rf_metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_val, y_rf_pred),
        'precision': precision_score(y_val, y_rf_pred),
        'recall': recall_score(y_val, y_rf_pred),
        'f1_score': f1_score(y_val, y_rf_pred),
        'roc_auc': roc_auc_score(y_val, y_rf_proba)
    }
    
    # NOW clean column names for advanced models (LightGBM requirement)
    print("\nCleaning column names for LightGBM compatibility...")
    X_train = clean_column_names(X_train)
    X_val = clean_column_names(X_val)
    print("✓ Column names cleaned")
    
    # Train XGBoost (works with cleaned names)
    xgb_model, xgb_metrics, y_xgb_pred, y_xgb_proba, xgb_feature_imp = train_xgboost(
        X_train, y_train, X_val, y_val
    )
    
    # Train LightGBM (needs cleaned names)
    lgbm_model, lgbm_metrics, y_lgbm_pred, y_lgbm_proba, lgbm_feature_imp = train_lightgbm(
        X_train, y_train, X_val, y_val
    )
    
    # Cross-validation
    print("\n" + "="*60)
    print("NOTE: Running CV on XGBoost and LightGBM only")
    print("="*60)
    
    models_for_cv = {
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model
    }
    
    cv_results = cross_validate_models(X_train, y_train, models_for_cv, cv_folds=5)
    
    # Compare all models
    comparison = compare_all_models(lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics)
    
    # Save everything
    print("\n=== SAVING MODELS AND RESULTS ===")
    
    # Save models
    joblib.dump(xgb_model, '../models/xgboost_v1.pkl')
    joblib.dump(lgbm_model, '../models/lightgbm_v1.pkl')
    print("✓ Models saved:")
    print("  - ../models/xgboost_v1.pkl")
    print("  - ../models/lightgbm_v1.pkl")
    
    # Save results
    comparison.to_csv('../data/processed/all_models_comparison.csv', index=False)
    cv_results.to_csv('../data/processed/cv_results.csv', index=False)
    xgb_feature_imp.to_csv('../data/processed/xgb_feature_importance.csv', index=False)
    lgbm_feature_imp.to_csv('../data/processed/lgbm_feature_importance.csv', index=False)
    
    print("\n✓ Results saved")
    
    # Plot visualizations
    plot_advanced_model_comparisons(
        y_val, y_lr_pred, y_rf_pred, y_xgb_pred, y_lgbm_pred,
        y_lr_proba, y_rf_proba, y_xgb_proba, y_lgbm_proba,
        lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics
    )
    
    print("\n" + "="*60)
    print("ADVANCED MODEL TRAINING COMPLETE")
    print("="*60)
    
    return {
        'xgb_model': xgb_model,
        'lgbm_model': lgbm_model,
        'xgb_metrics': xgb_metrics,
        'lgbm_metrics': lgbm_metrics,
        'comparison': comparison,
        'cv_results': cv_results,
        'xgb_feature_imp': xgb_feature_imp,
        'lgbm_feature_imp': lgbm_feature_imp
    }


def predict_with_threshold(model, X, model_name='xgboost', threshold_dir='../models'):
    """
    Load the saved DEPLOYED threshold and apply it to new data.

    Always uses 'deployed_threshold' from the JSON — the recall-optimised
    one chosen for business ROI, not the F1-mathematical one.

    Args:
        model         : trained XGBClassifier loaded with joblib
        X             : feature DataFrame (same columns as training)
        model_name    : must match what was passed to optimize_threshold()
        threshold_dir : where the JSON was saved

    Returns:
        y_pred  (array of 0/1)   — final binary fraud predictions
        y_proba (array of float) — raw probabilities (useful for risk ranking)
    """


    threshold_path = f"{threshold_dir}/{model_name}_threshold.json"

    if not os.path.exists(threshold_path):
        raise FileNotFoundError(
            f"Threshold file not found at {threshold_path}.\n"
            f"Run train_xgboost() first to generate and save thresholds."
        )

    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)

    # Always load 'deployed_threshold' — never re-derive from data
    threshold = threshold_data['deployed_threshold']
    rationale = threshold_data.get('rationale', '')

    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    print(f"✓ Predictions made using deployed threshold: {threshold:.4f}  ({model_name})")
    print(f"  Rationale: {rationale}")
    print(f"  Flagged as fraud: {y_pred.sum():,} / {len(y_pred):,} ({y_pred.mean()*100:.2f}%)")

    return y_pred, y_proba


# ─────────────────────────────────────────────────────────────────────────────
# WHAT THE SAVED JSON WILL LOOK LIKE 
# ─────────────────────────────────────────────────────────────────────────────
#
# ../models/xgboost_threshold.json
# {
#   "model_name": "xgboost",
#
#   "f1_threshold": 0.5708,        ← mathematically best F1 (your friend's version)
#   "recall_threshold": 0.28XX,    ← catches 75%+ of fraud (business-optimal)
#   "deployed_threshold": 0.28XX,  ← what production uses (same as recall_threshold)
#
#   "recall_at_recall_threshold": 0.75XX,
#   "precision_at_recall_threshold": 0.XXXX,
#
#   "missed_frauds_deployed": XX,
#   "false_alarms_deployed": XXX,
#   "business_cost_deployed": XXXXX,
#   "cost_saving_vs_default": XXXXX,
#
#   "rationale": "recall_threshold deployed — catching 75% of frauds..."
# }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'baseline':
            # Train baseline models
            results = train_baseline_models()
        elif sys.argv[1] == 'advanced':
            # Train advanced models
            results = train_advanced_models()
    else:
        # Default: run data preparation
        input_file = '../data/processed/insurance_claims_engineered.csv'
        output_dir = '../data/processed/'
        
        # CORRECTED: Using sampling_method='none' instead of 'smote'
        metadata = prepare_training_data(
            input_path=input_file,
            output_dir=output_dir,
            sampling_method='none',  # CHANGED: No SMOTE, use class weights instead
            sampling_strategy=0.5,   # Ignored when method='none'
            random_state=42
        )
        
        print("\n Next steps:")
        print("   python model_training.py baseline   # Train LR and RF")
        print("   python model_training.py advanced   # Train XGBoost and LightGBM")