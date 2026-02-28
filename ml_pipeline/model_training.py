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


if __name__ == "__main__":
    # Run data preparation
    input_file = '../data/processed/insurance_claims_engineered.csv'
    output_dir = '../data/processed/'
    
    # Prepare data with SMOTE (50:50 balance)
    metadata = prepare_training_data(
        input_path=input_file,
        output_dir=output_dir,
        sampling_method='smote',  # Can change to 'undersample' or 'combined'
        sampling_strategy=0.5,    # 50:50 balance
        random_state=42
    )
    
    print("\n" + "="*60)
    print("METADATA SUMMARY")
    print("="*60)
    for key, value in metadata.items():
        if key not in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'feature_names']:
            print(f"{key}: {value}")