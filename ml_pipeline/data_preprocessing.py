"""
Data Preprocessing Module
Cleans raw insurance claims data for model training
"""

import pandas as pd
import numpy as np
from typing import Tuple

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw data and create binary fraud target"""
    df = pd.read_csv(filepath)
    
    # Create binary fraud column if not exists
    if 'FraudFound_P' not in df.columns:
        df['FraudFound_P'] = (df['FraudFound'] == 'Yes').astype(int)
    
    print(f" Loaded {len(df):,} records")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in dataset"""
    df = df.copy()
    
    print("\n=== HANDLING MISSING VALUES ===")
    
    # Check for actual nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Found {null_counts.sum()} null values")
        print(null_counts[null_counts > 0])
    else:
        print("No null values found")
    
    # Handle Age = 0 (invalid)
    if (df['Age'] == 0).sum() > 0:
        print(f"\nFixing {(df['Age'] == 0).sum()} Age=0 values")
        # Replace 0 with median age
        median_age = df[df['Age'] > 0]['Age'].median()
        df.loc[df['Age'] == 0, 'Age'] = median_age
        print(f"Replaced with median: {median_age}")
    
    # Handle 'none' strings in categorical features
    # These represent "no value" but aren't nulls
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if 'none' in df[col].str.lower().unique():
            print(f"\n'{col}' has 'none' values - keeping as category")
            # Keep as valid category (represents "no past claims", etc.)
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records"""
    df = df.copy()
    
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    
    print(f"\n=== DUPLICATE REMOVAL ===")
    print(f"Removed {removed} duplicate rows")
    print(f"Remaining: {len(df):,} records")
    
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers in numerical features"""
    df = df.copy()
    
    print("\n=== OUTLIER HANDLING ===")
    
    # Identify numerical columns (exclude IDs and target)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols 
                      if col not in ['PolicyNumber', 'FraudFound_P', 'Year']]
    
    for col in numerical_cols:
        # Calculate percentiles
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        
        # Count outliers
        outliers_low = (df[col] < p1).sum()
        outliers_high = (df[col] > p99).sum()
        
        if outliers_low > 0 or outliers_high > 0:
            print(f"\n{col}:")
            print(f"  Low outliers (< {p1}): {outliers_low}")
            print(f"  High outliers (> {p99}): {outliers_high}")
            
            # Cap outliers at 1st and 99th percentile
            df[col] = df[col].clip(lower=p1, upper=p99)
            print(f"  ✓ Capped to range [{p1}, {p99}]")
    
    return df


def drop_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features not useful for modeling"""
    df = df.copy()
    
    print("\n=== DROPPING IRRELEVANT FEATURES ===")
    
    # Features to drop
    drop_cols = []
    
    # Drop ID columns (no predictive power)
    if 'PolicyNumber' in df.columns:
        drop_cols.append('PolicyNumber')
        print("✓ Dropping PolicyNumber (unique ID)")
    
    # Drop Year (no variation: only 1994-1996)
    if 'Year' in df.columns:
        drop_cols.append('Year')
        print("✓ Dropping Year (no variation)")
    
    # Drop original fraud column (we have binary version)
    if 'FraudFound' in df.columns:
        drop_cols.append('FraudFound')
        print("✓ Dropping FraudFound (using FraudFound_P)")
    
    # Drop RepNumber (representative ID, not predictive)
    if 'RepNumber' in df.columns:
        drop_cols.append('RepNumber')
        print("✓ Dropping RepNumber (rep assignment)")
    
    df = df.drop(columns=drop_cols, errors='ignore')
    
    print(f"\nRemaining features: {len(df.columns)}")
    print(f"Remaining samples: {len(df):,}")
    
    return df


def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types"""
    df = df.copy()
    
    print("\n=== DATA TYPE VALIDATION ===")
    
    # Force numeric columns to correct types
    numeric_cols = ['Age', 'WeekOfMonth', 'WeekOfMonthClaimed', 
                   'Deductible', 'DriverRating']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Force categorical columns to string type
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    print("✓ Data types validated")
    print(f"\nNumeric columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
    print(f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    return df


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Main preprocessing pipeline
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save cleaned CSV file
    
    Returns:
        Cleaned DataFrame
    """
    print("="*60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    df = load_raw_data(input_path)
    initial_shape = df.shape
    
    # Step 2: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    # Step 4: Drop irrelevant features
    df = drop_irrelevant_features(df)
    
    # Step 5: Handle outliers
    df = handle_outliers(df)
    
    # Step 6: Validate data types
    df = validate_data_types(df)
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Initial shape: {initial_shape}")
    print(f"Final shape: {df.shape}")
    print(f"Records removed: {initial_shape[0] - df.shape[0]}")
    print(f"Features removed: {initial_shape[1] - df.shape[1]}")
    print(f"\nFraud distribution:")
    print(df['FraudFound_P'].value_counts())
    print(f"Fraud rate: {df['FraudFound_P'].mean()*100:.2f}%")
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\n Cleaned data saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Run preprocessing
    input_file = '../data/raw/insurance_claims_raw.csv'
    output_file = '../data/processed/insurance_claims_cleaned.csv'
    
    df_clean = preprocess_data(input_file, output_file)
    
    print("\n" + "="*60)
    print("Preview of cleaned data:")
    print("="*60)
    print(df_clean.head())
    print(f"\nColumns: {list(df_clean.columns)}")