"""
Model Training Module
Handles train-test split, class imbalance, and model training preparation
"""

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
    
    print(f" Loaded engineered data")
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
                           method: str = 'smote',
                           sampling_strategy: float = 0.5,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE or combined sampling
    
    Args:
        X_train: Training features
        y_train: Training target
        method: 'smote', 'undersample', or 'combined'
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
    
    # Apply sampling method
    if method == 'smote':
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
        raise ValueError(f"Unknown method: {method}. Use 'smote', 'undersample', or 'combined'")
    
    # Calculate new distribution
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
    
    print(f"\n No data leakage - train/val/test split before SMOTE guarantees independence")
    
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
    
    print("\n Data splits verified and ready for modeling")


def prepare_training_data(input_path: str,
                         output_dir: str = '../data/processed/',
                         sampling_method: str = 'smote',
                         sampling_strategy: float = 0.5,
                         random_state: int = 42) -> dict:
    """
    Complete data preparation pipeline for model training
    
    Args:
        input_path: Path to engineered CSV
        output_dir: Directory to save split datasets
        sampling_method: 'smote', 'undersample', or 'combined'
        sampling_strategy: Target minority/majority ratio
        random_state: Random seed
    
    Returns:
        Dictionary with all splits and metadata
    """
    print("="*60)
    print("PREPARING DATA FOR MODEL TRAINING")
    print("="*60)
    
    # Step 1: Load engineered data
    X, y = load_engineered_data(input_path)
    
    # Step 2: Split into train, val, test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=random_state)
    
    # Step 3: Handle class imbalance (ONLY on training set)
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train, y_train,
        method=sampling_method,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    # Step 4: Verify everything
    verify_splits(X_train_balanced, X_val, X_test, y_train_balanced, y_val, y_test)
    
    # Step 5: Save all splits
    print("\n=== SAVING DATASETS ===")
    
    # Save training data (balanced)
    train_data = pd.concat([X_train_balanced, y_train_balanced], axis=1)
    train_path = f"{output_dir}/train_balanced.csv"
    train_data.to_csv(train_path, index=False)
    print(f"✓ Training (balanced): {train_path}")
    
    # Save validation data (unbalanced - realistic)
    val_data = pd.concat([X_val, y_val], axis=1)
    val_path = f"{output_dir}/validation.csv"
    val_data.to_csv(val_path, index=False)
    print(f"✓ Validation: {val_path}")
    
    # Save test data (unbalanced - realistic)
    test_data = pd.concat([X_test, y_test], axis=1)
    test_path = f"{output_dir}/test.csv"
    test_data.to_csv(test_path, index=False)
    print(f"✓ Test: {test_path}")
    
    # Create metadata
    metadata = {
        'X_train': X_train_balanced,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train_balanced,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'n_features': X.shape[1],
        'train_samples': len(X_train_balanced),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'original_fraud_rate': y.mean(),
        'train_fraud_rate': y_train_balanced.mean(),
        'val_fraud_rate': y_val.mean(),
        'test_fraud_rate': y_test.mean(),
    }
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Training samples: {metadata['train_samples']:,}")
    print(f"Validation samples: {metadata['val_samples']:,}")
    print(f"Test samples: {metadata['test_samples']:,}")
    print(f"Features: {metadata['n_features']}")
    print("\n Ready for model training!")
    
    return metadata


def train_logistic_regression(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train Logistic Regression baseline model
    
    Args:
        X_train, y_train: Training data (balanced)
        X_val, y_val: Validation data (unbalanced)
        random_state: Random seed
    
    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION BASELINE")
    print("="*60)
    
    # Train model
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,  # Increase for convergence
        solver='lbfgs'
    )
    
    lr_model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Predict on validation set
    y_val_pred = lr_model.predict(X_val)
    y_val_proba = lr_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Print metrics
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Legitimate', 'Fraud']))
    
    return lr_model, metrics, y_val_pred, y_val_proba


def train_random_forest(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train Random Forest baseline model
    
    Args:
        X_train, y_train: Training data (balanced)
        X_val, y_val: Validation data (unbalanced)
        random_state: Random seed
    
    Returns:
        Trained model, metrics, and feature importances
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST BASELINE")
    print("="*60)
    
    # Train model
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Default
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    rf_model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Predict on validation set
    y_val_pred = rf_model.predict(X_val)
    y_val_proba = rf_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Print metrics
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Legitimate', 'Fraud']))
    
    # Feature Importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10).to_string(index=False))
    
    return rf_model, metrics, y_val_pred, y_val_proba, feature_importance


def compare_baseline_models(lr_metrics, rf_metrics):
    """Compare performance of baseline models"""
    print("\n" + "="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame([lr_metrics, rf_metrics])
    comparison = comparison[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    
    print("\n")
    print(comparison.to_string(index=False))
    
    # Identify best model
    best_f1_idx = comparison['f1_score'].idxmax()
    best_model = comparison.loc[best_f1_idx, 'model']
    best_f1 = comparison.loc[best_f1_idx, 'f1_score']
    
    print(f"\n Best Baseline Model: {best_model}")
    print(f"   F1-Score: {best_f1:.4f}")
    
    return comparison


def plot_confusion_matrices(y_val, y_lr_pred, y_rf_pred, save_path='../data/processed/'):
    """Plot confusion matrices for both baseline models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Logistic Regression
    cm_lr = confusion_matrix(y_val, y_lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[0].set_title('Logistic Regression\nConfusion Matrix', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Random Forest
    cm_rf = confusion_matrix(y_val, y_rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[1].set_title('Random Forest\nConfusion Matrix', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/baseline_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to {save_path}/baseline_confusion_matrices.png")
    plt.show()


def plot_roc_curves(y_val, y_lr_proba, y_rf_proba, lr_auc, rf_auc, save_path='../data/processed/'):
    """Plot ROC curves for both baseline models"""
    plt.figure(figsize=(10, 6))
    
    # Logistic Regression ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_val, y_lr_proba)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.4f})', 
             linewidth=2, color='blue')
    
    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_val, y_rf_proba)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.4f})', 
             linewidth=2, color='green')
    
    # Diagonal (random baseline)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Baseline Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/baseline_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved to {save_path}/baseline_roc_curves.png")
    plt.show()


def train_baseline_models():
    """Complete baseline model training pipeline"""
    print("="*60)
    print("BASELINE MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load prepared data
    print("\nLoading prepared datasets...")
    train_df = pd.read_csv('../data/processed/train_balanced.csv')
    val_df = pd.read_csv('../data/processed/validation.csv')
    
    X_train = train_df.drop('FraudFound_P', axis=1)
    y_train = train_df['FraudFound_P']
    X_val = val_df.drop('FraudFound_P', axis=1)
    y_val = val_df['FraudFound_P']
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Validation data: {X_val.shape}")
    
    # Train Logistic Regression
    lr_model, lr_metrics, y_lr_pred, y_lr_proba = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    
    # Train Random Forest
    rf_model, rf_metrics, y_rf_pred, y_rf_proba, rf_feature_imp = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    
    # Compare models
    comparison = compare_baseline_models(lr_metrics, rf_metrics)
    
    # Save comparison table
    comparison.to_csv('../data/processed/baseline_comparison.csv', index=False)
    print("\n✓ Comparison table saved to ../data/processed/baseline_comparison.csv")
    
    # Plot visualizations
    plot_confusion_matrices(y_val, y_lr_pred, y_rf_pred)
    plot_roc_curves(y_val, y_lr_proba, y_rf_proba, lr_metrics['roc_auc'], rf_metrics['roc_auc'])
    
    # Save feature importances
    rf_feature_imp.to_csv('../data/processed/rf_feature_importance.csv', index=False)
    print("\n✓ Feature importances saved to ../data/processed/rf_feature_importance.csv")
    
    # Save models
    joblib.dump(lr_model, '../models/logistic_regression_baseline.pkl')
    joblib.dump(rf_model, '../models/random_forest_baseline.pkl')
    print("\n✓ Models saved:")
    print("  - ../models/logistic_regression_baseline.pkl")
    print("  - ../models/random_forest_baseline.pkl")
    
    print("\n" + "="*60)
    print("BASELINE TRAINING COMPLETE")
    print("="*60)
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'comparison': comparison,
        'rf_feature_importance': rf_feature_imp
    }


def calculate_scale_pos_weight(y_train):
    """Calculate class weight for imbalanced data"""
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    scale_pos_weight = legit_count / fraud_count
    
    print(f"\n=== CLASS WEIGHT CALCULATION ===")
    print(f"Legitimate samples: {legit_count:,}")
    print(f"Fraud samples: {fraud_count:,}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    print(f"Interpretation: Fraud is {scale_pos_weight:.1f}x less common than legitimate")
    
    return scale_pos_weight


def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight=None, random_state=42):
    """
    Train XGBoost model with proper fraud detection configuration
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST CLASSIFIER")
    print("="*60)
    
    # CRITICAL: Use original class imbalance ratio (before SMOTE)
    # Not the SMOTE-balanced ratio!
    if scale_pos_weight is None:
        # Calculate from original data ratio: ~6% fraud = 15.7:1 ratio
        scale_pos_weight = 15.7  # Hardcoded based on original 6% fraud rate
        
        print(f"\n=== CLASS WEIGHT CONFIGURATION ===")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        print(f"Based on original fraud rate: ~6%")
        print(f"Interpretation: Penalize missing fraud 15.7x more than false alarm")
    
    # Configure XGBoost for fraud detection
    print("\nTraining XGBoost with fraud-optimized parameters...")
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,  # CRITICAL for imbalanced data
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print("✓ Model trained")
    
    # Predict on validation set
    y_val_pred = xgb_model.predict(X_val)
    y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': 'XGBoost',
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Print metrics
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f} (fraud detection rate)")
    print(f"F1-Score:  {metrics['f1_score']:.4f}  PRIMARY METRIC")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    # Business interpretation
    fraud_caught = cm[1,1]
    fraud_total = cm[1,1] + cm[1,0]
    fraud_caught_pct = (fraud_caught / fraud_total) * 100
    
    print(f"\n💡 Business Impact:")
    print(f"   Caught {fraud_caught}/{fraud_total} frauds ({fraud_caught_pct:.1f}%)")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Legitimate', 'Fraud']))
    
    # Feature Importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10).to_string(index=False))
    
    return xgb_model, metrics, y_val_pred, y_val_proba, feature_importance

def clean_column_names(df):
    """Remove special characters from column names for LightGBM"""
    df = df.copy()
    df.columns = df.columns.str.replace(':', '_').str.replace('-', '_').str.replace(' ', '_')
    return df


def train_lightgbm(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train LightGBM model with is_unbalance parameter
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        random_state: Random seed
    
    Returns:
        Trained model, metrics, predictions, and feature importances
    """
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM CLASSIFIER")
    print("="*60)
    
    # Configure LightGBM for fraud detection
    print("\nTraining LightGBM with is_unbalance=True...")
    lgbm_model = LGBMClassifier(
        is_unbalance=True,                 # Handle class imbalance (LightGBM's approach)
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1                         # Suppress training logs
    )
    
    # Train model
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print("✓ Model trained")
    
    # Predict on validation set
    y_val_pred = lgbm_model.predict(X_val)
    y_val_proba = lgbm_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': 'LightGBM',
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Print metrics
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ⭐ PRIMARY METRIC")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    # Business interpretation
    fraud_caught = cm[1,1]
    fraud_total = cm[1,1] + cm[1,0]
    fraud_caught_pct = (fraud_caught / fraud_total) * 100
    
    print(f"\n💡 Business Impact:")
    print(f"   Caught {fraud_caught}/{fraud_total} frauds ({fraud_caught_pct:.1f}%)")
    print(f"   Missed {cm[1,0]} frauds (cost: ${cm[1,0] * 10000:,})")
    print(f"   False alarms: {cm[0,1]} (cost: ${cm[0,1] * 200:,})")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Legitimate', 'Fraud']))
    
    # Feature Importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgbm_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10).to_string(index=False))
    
    return lgbm_model, metrics, y_val_pred, y_val_proba, feature_importance


def cross_validate_models(X_train, y_train, models_dict, cv_folds=5):
    """
    Perform stratified cross-validation on multiple models
    
    Args:
        X_train, y_train: Training data
        models_dict: Dictionary of {name: model}
        cv_folds: Number of CV folds
    
    Returns:
        DataFrame with CV results
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-FOLD STRATIFIED)")
    print("="*60)
    
    cv_results = []
    
    # Set up stratified K-fold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model_name, model in models_dict.items():
        print(f"\nRunning CV for {model_name}...")
        
        # Run cross-validation for multiple metrics
        f1_scores = cross_val_score(model, X_train, y_train, cv=skf, 
                                     scoring='f1', n_jobs=-1)
        recall_scores = cross_val_score(model, X_train, y_train, cv=skf, 
                                        scoring='recall', n_jobs=-1)
        precision_scores = cross_val_score(model, X_train, y_train, cv=skf, 
                                          scoring='precision', n_jobs=-1)
        
        result = {
            'model': model_name,
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'recall_mean': recall_scores.mean(),
            'recall_std': recall_scores.std(),
            'precision_mean': precision_scores.mean(),
            'precision_std': precision_scores.std()
        }
        
        cv_results.append(result)
        
        print(f"  F1-Score:  {result['f1_mean']:.4f} (±{result['f1_std']:.4f})")
        print(f"  Recall:    {result['recall_mean']:.4f} (±{result['recall_std']:.4f})")
        print(f"  Precision: {result['precision_mean']:.4f} (±{result['precision_std']:.4f})")
    
    cv_df = pd.DataFrame(cv_results)
    
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(cv_df.to_string(index=False))
    
    return cv_df


def compare_all_models(lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics):
    """Compare all 4 models"""
    print("\n" + "="*60)
    print("COMPLETE MODEL COMPARISON (ALL 4 MODELS)")
    print("="*60)
    
    comparison = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics])
    comparison = comparison[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    
    print("\n")
    print(comparison.to_string(index=False))
    
    # Identify best models
    best_f1_idx = comparison['f1_score'].idxmax()
    best_recall_idx = comparison['recall'].idxmax()
    best_auc_idx = comparison['roc_auc'].idxmax()
    
    print(f"\n🏆 WINNERS:")
    print(f"   Best F1-Score: {comparison.loc[best_f1_idx, 'model']} ({comparison.loc[best_f1_idx, 'f1_score']:.4f})")
    print(f"   Best Recall: {comparison.loc[best_recall_idx, 'model']} ({comparison.loc[best_recall_idx, 'recall']:.4f})")
    print(f"   Best ROC-AUC: {comparison.loc[best_auc_idx, 'model']} ({comparison.loc[best_auc_idx, 'roc_auc']:.4f})")
    
    # Calculate improvement over baseline
    baseline_f1 = comparison[comparison['model'].str.contains('Logistic')]['f1_score'].values[0]
    best_f1 = comparison.loc[best_f1_idx, 'f1_score']
    improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
    
    print(f"\n📈 IMPROVEMENT OVER BASELINE:")
    print(f"   F1-Score improved by {improvement:.1f}%")
    print(f"   {baseline_f1:.4f} → {best_f1:.4f}")
    
    return comparison


def plot_advanced_model_comparisons(y_val, y_lr_pred, y_rf_pred, y_xgb_pred, y_lgbm_pred,
                                    y_lr_proba, y_rf_proba, y_xgb_proba, y_lgbm_proba,
                                    lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics,
                                    save_path='../data/processed/'):
    """Plot comprehensive model comparisons"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrices (2x2 grid)
    cms = [
        confusion_matrix(y_val, y_lr_pred),
        confusion_matrix(y_val, y_rf_pred),
        confusion_matrix(y_val, y_xgb_pred),
        confusion_matrix(y_val, y_lgbm_pred)
    ]
    titles = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
    cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']
    
    for idx, (cm, title, cmap) in enumerate(zip(cms, titles, cmaps)):
        ax = axes[idx // 2, idx % 2]
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax.set_title(f'{title}\nF1: {[lr_metrics, rf_metrics, xgb_metrics, lgbm_metrics][idx]["f1_score"]:.4f}', 
                    fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved")
    plt.close()
    
    # 2. ROC Curves
    plt.figure(figsize=(10, 6))
    
    models_data = [
        ('Logistic Regression', y_lr_proba, lr_metrics['roc_auc'], 'blue'),
        ('Random Forest', y_rf_proba, rf_metrics['roc_auc'], 'green'),
        ('XGBoost', y_xgb_proba, xgb_metrics['roc_auc'], 'orange'),
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
    train_df = pd.read_csv('../data/processed/train_balanced.csv')
    val_df = pd.read_csv('../data/processed/validation.csv')
    
    X_train = train_df.drop('FraudFound_P', axis=1)
    y_train = train_df['FraudFound_P']
    X_val = val_df.drop('FraudFound_P', axis=1)
    y_val = val_df['FraudFound_P']
    
    print(f"✓ Training: {X_train.shape}")
    print(f"✓ Validation: {X_val.shape}")
    
    # Load baseline results for comparison (BEFORE cleaning column names)
    print("\nLoading baseline models...")
    lr_model = joblib.load('../models/logistic_regression_baseline.pkl')
    rf_model = joblib.load('../models/random_forest_baseline.pkl')
    
    # Get baseline metrics (with ORIGINAL column names)
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
        
        metadata = prepare_training_data(
            input_path=input_file,
            output_dir=output_dir,
            sampling_method='smote',
            sampling_strategy=0.5,
            random_state=42
        )
        
        print("\n💡 Next steps:")
        print("   python model_training.py baseline   # Train LR and RF")
        print("   python model_training.py advanced   # Train XGBoost and LightGBM")