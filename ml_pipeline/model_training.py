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
    print(" Model trained")
    
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
    print(" Model trained")
    
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


if __name__ == "__main__":
    import sys
    
    # Check command line argument
    if len(sys.argv) > 1 and sys.argv[1] == 'baseline':
        # Train baseline models
        results = train_baseline_models()
    else:
        # Run data preparation (Day 8 task)
        input_file = '../data/processed/insurance_claims_engineered.csv'
        output_dir = '../data/processed/'
        
        metadata = prepare_training_data(
            input_path=input_file,
            output_dir=output_dir,
            sampling_method='smote',
            sampling_strategy=0.5,
            random_state=42
        )
        
        print("\n" + "="*60)
        print("METADATA SUMMARY")
        print("="*60)
        for key, value in metadata.items():
            if key not in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'feature_names']:
                print(f"{key}: {value}")
        
        print("\n💡 To train baseline models, run:")
        print("   python model_training.py baseline")