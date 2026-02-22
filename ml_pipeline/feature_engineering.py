"""
Feature Engineering Module
Creates model-ready features from cleaned insurance data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Tuple, Dict

def create_ordinal_mappings() -> Dict[str, Dict]:
    """
    Define ordinal mappings for categorical features with natural ordering
    Returns dictionary of {column_name: {value: numeric_code}}
    """
    
    ordinal_mappings = {
        'AgeOfVehicle': {
            'new': 0,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            'more than 7': 8
        },
        
        'VehiclePrice': {
            'less than 20000': 1,
            '20000 to 29000': 2,
            '30000 to 39000': 3,
            '40000 to 59000': 4,
            '60000 to 69000': 5,
            'more than 69000': 6
        },
        
        'AgeOfPolicyHolder': {
            '16 to 17': 1,
            '18 to 20': 2,
            '21 to 25': 3,
            '26 to 30': 4,
            '31 to 35': 5,
            '36 to 40': 6,
            '41 to 50': 7,
            '51 to 65': 8,
            'over 65': 9
        },
        
        'PastNumberOfClaims': {
            'none': 0,
            '1': 1,
            '2 to 4': 2,
            'more than 4': 3
        },
        
        'Days:Policy-Accident': {
            'none': 0,
            '1 to 7': 1,
            '8 to 15': 2,
            '15 to 30': 3,
            'more than 30': 4
        },
        
        'Days:Policy-Claim': {
            'none': 0,
            '8 to 15': 1,
            '15 to 30': 2,
            'more than 30': 3
        },
        
        'NumberOfSuppliments': {
            'none': 0,
            '1 to 2': 1,
            '3 to 5': 2,
            'more than 5': 3
        },
        
        'AddressChange-Claim': {
            'no change': 0,
            '1 year': 1,
            '2 to 3 years': 2,
            '4 to 8 years': 3,
            'under 6 months': 4  # Most suspicious
        },
        
        'NumberOfCars': {
            '1 vehicle': 1,
            '2 vehicles': 2,
            '3 to 4': 3,
            '5 to 8': 4,
            'more than 8': 5
        }
    }
    
    return ordinal_mappings


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ordinal categorical features to numeric scales"""
    df = df.copy()
    
    print("\n=== ENCODING ORDINAL FEATURES ===")
    
    mappings = create_ordinal_mappings()
    
    for col, mapping in mappings.items():
        if col in df.columns:
            # Handle any values not in mapping (assign max+1) VERY IMPORTANT to avoid NaNs
            # IT ALSO PRINTS A WARNING IF UNMAPPED VALUES ARE FOUND, SO I CAN KNOW IT HAPPENED
            unknown_values = set(df[col].unique()) - set(mapping.keys())
            if unknown_values:
                print(f"  Warning: {col} has unmapped values: {unknown_values}")
                max_val = max(mapping.values())
                for val in unknown_values:
                    mapping[val] = max_val + 1
            
            df[col + '_encoded'] = df[col].map(mapping)
            print(f"✓ {col}: {len(mapping)} categories → numeric scale")
    
    return df


def create_domain_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create fraud indicator flags based on domain knowledge"""
    df = df.copy()
    
    print("\n=== CREATING DOMAIN-SPECIFIC FLAGS ===")
    
    # Critical fraud signals
    df['no_police_report'] = (df['PoliceReportFiled'] == 'No').astype(int)
    df['no_witness'] = (df['WitnessPresent'] == 'No').astype(int)
    df['policy_holder_fault'] = (df['Fault'] == 'Policy Holder').astype(int)
    df['external_agent'] = (df['AgentType'] == 'External').astype(int)
    
    # Temporal flags
    df['quick_claim'] = (df['Days:Policy-Accident'].isin(['none', '1 to 7'])).astype(int)
    df['very_quick_claim_filing'] = (df['Days:Policy-Claim'] == 'none').astype(int)
    
    # History flags
    df['has_past_claims'] = (df['PastNumberOfClaims'] != 'none').astype(int)
    df['multiple_past_claims'] = (df['PastNumberOfClaims'].isin(['2 to 4', 'more than 4'])).astype(int)
    
    # Other suspicious patterns
    df['recent_address_change'] = (df['AddressChange-Claim'] == 'under 6 months').astype(int)
    df['many_supplements'] = (df['NumberOfSuppliments'].isin(['3 to 5', 'more than 5'])).astype(int)
    df['urban_accident'] = (df['AccidentArea'] == 'Urban').astype(int)
    
    print(f"✓ Created 11 domain-specific flags")
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features combining multiple signals"""
    df = df.copy()
    
    print("\n=== CREATING INTERACTION FEATURES ===")
    
    # High-risk combination: no police + no witness
    df['high_risk_combo'] = df['no_police_report'] + df['no_witness']  # 0, 1, or 2
    
    # Agent + Fault interaction
    df['external_agent_holder_fault'] = df['external_agent'] * df['policy_holder_fault']
    
    # Quick claim + No police
    df['quick_claim_no_police'] = df['quick_claim'] * df['no_police_report']
    
    # Past claims + Quick current claim
    df['repeat_claimer_quick'] = df['has_past_claims'] * df['quick_claim']
    
    # Urban + No witness
    df['urban_no_witness'] = df['urban_accident'] * df['no_witness']
    
    print(f"✓ Created 5 interaction features")
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features"""
    df = df.copy()
    
    print("\n=== CREATING TEMPORAL FEATURES ===")
    
    # Weekend flag
    weekend_days = ['Saturday', 'Sunday']
    df['is_weekend'] = df['DayOfWeek'].isin(weekend_days).astype(int)
    
    # End of year flag (Nov-Dec often have more claims)
    df['is_end_of_year'] = df['Month'].isin(['Nov', 'Dec']).astype(int)
    
    # Claim filed same month as accident
    df['claim_same_month'] = (df['Month'] == df['MonthClaimed']).astype(int)
    
    # Map month to numeric
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month_numeric'] = df['Month'].map(month_map)
    df['MonthClaimed_numeric'] = df['MonthClaimed'].map(month_map)
    
    # Day of week numeric (Monday=0, Sunday=6)
    day_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df['DayOfWeek_numeric'] = df['DayOfWeek'].map(day_map)
    df['DayOfWeekClaimed_numeric'] = df['DayOfWeekClaimed'].map(day_map)
    
    print(f"✓ Created 7 temporal features")
    
    return df


def encode_binary_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary categorical features"""
    df = df.copy()
    
    print("\n=== ENCODING BINARY CATEGORICALS ===")
    
    binary_mappings = {
        'PoliceReportFiled': {'No': 0, 'Yes': 1},
        'WitnessPresent': {'No': 0, 'Yes': 1},
        'Fault': {'Third Party': 0, 'Policy Holder': 1},
        'AgentType': {'Internal': 0, 'External': 1},
        'AccidentArea': {'Rural': 0, 'Urban': 1},
        'Sex': {'Female': 0, 'Male': 1}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col + '_binary'] = df[col].map(mapping)
            print(f"✓ {col}: {list(mapping.keys())} → binary")
    
    return df


def one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categorical features"""
    df = df.copy()
    
    print("\n=== ONE-HOT ENCODING CATEGORICALS ===")
    
    # Features to one-hot encode
    ohe_features = [
        'Make',          # Vehicle manufacturer
        'MaritalStatus', # Single, Married, etc.
        'PolicyType',    # Type of policy
        'VehicleCategory', # Sport, Sedan, Utility
        'BasePolicy'     # Liability, Collision, All Perils
    ]
    
    for col in ohe_features:
        if col in df.columns:
            # Get dummies with drop_first to avoid multicollinearity
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            print(f"✓ {col}: {df[col].nunique()} categories → {len(dummies.columns)} features")
    
    return df


def scale_numerical_features(df: pd.DataFrame, 
                            scaler: StandardScaler = None,
                            fit: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler"""
    df = df.copy()
    
    print("\n=== SCALING NUMERICAL FEATURES ===")
    
    # Numerical features to scale
    numerical_cols = ['Age', 'Deductible', 'DriverRating', 'WeekOfMonth', 'WeekOfMonthClaimed']
    
    # Also scale encoded ordinal features
    ordinal_encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
    numerical_cols.extend(ordinal_encoded_cols)
    
    # Filter to existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if fit:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print(f"✓ Fitted and transformed {len(numerical_cols)} numerical features")
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        print(f"✓ Transformed {len(numerical_cols)} numerical features")
    
    return df, scaler


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select final feature set for modeling"""
    df = df.copy()
    
    print("\n=== SELECTING FINAL FEATURES ===")
    
    # Drop original categorical columns (we have encoded versions)
    drop_cols = [
        # Original ordinal features (we have _encoded versions)
        'AgeOfVehicle', 'VehiclePrice', 'AgeOfPolicyHolder', 'PastNumberOfClaims',
        'Days:Policy-Accident', 'Days:Policy-Claim', 'NumberOfSuppliments',
        'AddressChange-Claim', 'NumberOfCars',
        
        # Original binary features (we have _binary versions)
        'PoliceReportFiled', 'WitnessPresent', 'Fault', 'AgentType', 'AccidentArea', 'Sex',
        
        # Original nominal features (we have one-hot encoded)
        'Make', 'MaritalStatus', 'PolicyType', 'VehicleCategory', 'BasePolicy',
        
        # String temporal features (we have numeric versions)
        'Month', 'DayOfWeek', 'MonthClaimed', 'DayOfWeekClaimed'
    ]
    
    # Drop columns that exist
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols)
    
    print(f"✓ Dropped {len(drop_cols)} original categorical columns")
    print(f"✓ Remaining features: {len(df.columns) - 1} (excluding target)")  # -1 for FraudFound_P
    
    return df


def engineer_features(input_path: str, 
                      output_path: str,
                      scaler_path: str = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Main feature engineering pipeline
    
    Args:
        input_path: Path to cleaned CSV
        output_path: Path to save engineered CSV
        scaler_path: Path to save scaler (optional)
    
    Returns:
        Engineered DataFrame and fitted scaler
    """
    print("="*60)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load cleaned data
    df = pd.read_csv(input_path)
    print(f"\n Loaded {len(df):,} records with {len(df.columns)} features")
    initial_cols = len(df.columns)
    
    # Step 1: Encode ordinal features
    df = encode_ordinal_features(df)
    
    # Step 2: Create domain-specific flags
    df = create_domain_flags(df)
    
    # Step 3: Create interaction features
    df = create_interaction_features(df)
    
    # Step 4: Create temporal features
    df = create_temporal_features(df)
    
    # Step 5: Encode binary categoricals
    df = encode_binary_categoricals(df)
    
    # Step 6: One-hot encode nominal categoricals
    df = one_hot_encode_categoricals(df)
    
    # Step 7: Scale numerical features
    df, scaler = scale_numerical_features(df, fit=True)
    
    # Step 8: Select final features
    df = select_final_features(df)
    
    # Final summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Initial features: {initial_cols}")
    print(f"Final features: {len(df.columns)} (including target)")
    print(f"Feature columns: {len(df.columns) - 1}")
    print(f"Records: {len(df):,}")
    print(f"\nFraud distribution:")
    print(df['FraudFound_P'].value_counts())
    
    # Save engineered data
    df.to_csv(output_path, index=False)
    print(f"\n Engineered data saved to: {output_path}")
    
    # Save scaler
    if scaler_path:
        joblib.dump(scaler, scaler_path)
        print(f" Scaler saved to: {scaler_path}")
    
    return df, scaler


if __name__ == "__main__":
    # Run feature engineering
    input_file = '../data/processed/insurance_claims_cleaned.csv'
    output_file = '../data/processed/insurance_claims_engineered.csv'
    scaler_file = '../models/scaler_v1.pkl'
    
    df_engineered, scaler = engineer_features(input_file, output_file, scaler_file)
    
    print("\n" + "="*60)
    print("Preview of engineered features:")
    print("="*60)
    print(df_engineered.head())
    
    print("\n" + "="*60)
    print("Final feature list:")
    print("="*60)
    feature_cols = [col for col in df_engineered.columns if col != 'FraudFound_P']
    for i, col in enumerate(feature_cols, 1):
        print(f"{i}. {col}")