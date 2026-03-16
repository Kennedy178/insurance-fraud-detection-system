# app/services/feature_service.py  (definitive fix — Day 17)
"""
Feature Engineering Service
=============================
Replicates feature_engineering.py exactly for API inference.
All string columns are explicitly cast to str before comparisons
to prevent .str accessor / dtype errors on single-row DataFrames.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger
from app.core.config import settings


class FeatureService:

    def __init__(self):
        self.scaler         = None
        self.feature_names  = None
        self._scaler_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        scaler_path   = os.path.join(settings.MODEL_DIR, 'scaler_v1.pkl')
        metadata_path = os.path.join(settings.MODEL_DIR, 'feature_metadata.json')

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self._scaler_loaded = True
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"scaler_v1.pkl not found at {scaler_path}")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata['feature_names']
            logger.info(f"Feature names loaded: {len(self.feature_names)} features")

    # ── MAIN ENTRY POINT ──────────────────────────────────────────────────

    def transform(self, raw_claim: Dict) -> pd.DataFrame:
        # Ensure all values are plain Python types — no enums, no objects
        raw_claim = {
            k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
            for k, v in raw_claim.items()
        }

        df = pd.DataFrame([raw_claim])

        # Force all object columns to str dtype explicitly
        # This is the definitive fix for the .str accessor error
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)

        df = self._encode_ordinal_features(df)
        df = self._create_domain_flags(df)
        df = self._create_interaction_features(df)
        df = self._create_temporal_features(df)
        df = self._encode_binary_categoricals(df)
        df = self._one_hot_encode_categoricals(df)
        df = self._scale_numerical_features(df)
        df = self._select_final_features(df)
        return df

    # ── HELPER ────────────────────────────────────────────────────────────

    def _get(self, df: pd.DataFrame, col: str, default: str) -> pd.Series:
        """
        Safe column accessor.
        Returns column as explicit str Series if present,
        otherwise a Series filled with the default string value.
        Always returns strings — never raises .str accessor errors.
        """
        if col in df.columns:
            return df[col].astype(str)
        return pd.Series([str(default)] * len(df), index=df.index)

    # ── STEP 1: ORDINAL ENCODING ──────────────────────────────────────────

    def _encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mappings = {
            'AgeOfVehicle': {
                'new': 0, '2 years': 2, '3 years': 3, '4 years': 4,
                '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 8
            },
            'VehiclePrice': {
                'less than 20,000': 1, '20,000 to 29,000': 2,
                '30,000 to 39,000': 3, '40,000 to 59,000': 4,
                '60,000 to 69,000': 5, 'more than 69,000': 6
            },
            'AgeOfPolicyHolder': {
                '16 to 17': 1, '18 to 20': 2, '21 to 25': 3,
                '26 to 30': 4, '31 to 35': 5, '36 to 40': 6,
                '41 to 50': 7, '51 to 65': 8, 'over 65': 9
            },
            'PastNumberOfClaims': {
                'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3
            },
            'Days:Policy-Accident': {
                'none': 0, '1 to 7': 1, '8 to 15': 2,
                '15 to 30': 3, 'more than 30': 4
            },
            'Days:Policy-Claim': {
                'none': 0, '8 to 15': 1, '15 to 30': 2, 'more than 30': 3
            },
            'NumberOfSuppliments': {
                'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3
            },
            'AddressChange-Claim': {
                'no change': 0, '1 year': 1, '2 to 3 years': 2,
                '4 to 8 years': 3, 'under 6 months': 4
            },
            'NumberOfCars': {
                '1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3,
                '5 to 8': 4, 'more than 8': 5
            },
        }

        for col, mapping in mappings.items():
            series = self._get(df, col, 'none')
            unknown = set(series.unique()) - set(mapping.keys()) - {'nan', 'None'}
            if unknown:
                max_val = max(mapping.values())
                for val in unknown:
                    mapping[val] = max_val + 1
            df[col + '_encoded'] = series.map(mapping).fillna(0).astype(float)

        return df

    # ── STEP 2: DOMAIN FLAGS ──────────────────────────────────────────────

    def _create_domain_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        police  = self._get(df, 'PoliceReportFiled', 'No')
        witness = self._get(df, 'WitnessPresent', 'No')
        fault   = self._get(df, 'Fault', 'Third Party')
        agent   = self._get(df, 'AgentType', 'Internal')
        days_pa = self._get(df, 'Days:Policy-Accident', 'more than 30')
        days_pc = self._get(df, 'Days:Policy-Claim', 'more than 30')
        past    = self._get(df, 'PastNumberOfClaims', 'none')
        addr    = self._get(df, 'AddressChange-Claim', 'no change')
        supps   = self._get(df, 'NumberOfSuppliments', 'none')
        area    = self._get(df, 'AccidentArea', 'Rural')

        df['no_police_report']        = (police == 'No').astype(int)
        df['no_witness']              = (witness == 'No').astype(int)
        df['policy_holder_fault']     = (fault == 'Policy Holder').astype(int)
        df['external_agent']          = (agent == 'External').astype(int)
        df['quick_claim']             = days_pa.isin(['none', '1 to 7']).astype(int)
        df['very_quick_claim_filing'] = (days_pc == 'none').astype(int)
        df['has_past_claims']         = (past != 'none').astype(int)
        df['multiple_past_claims']    = past.isin(['2 to 4', 'more than 4']).astype(int)
        df['recent_address_change']   = (addr == 'under 6 months').astype(int)
        df['many_supplements']        = supps.isin(['3 to 5', 'more than 5']).astype(int)
        df['urban_accident']          = (area == 'Urban').astype(int)

        return df

    # ── STEP 3: INTERACTION FEATURES ─────────────────────────────────────

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['high_risk_combo']             = df['no_police_report'] + df['no_witness']
        df['external_agent_holder_fault'] = df['external_agent'] * df['policy_holder_fault']
        df['quick_claim_no_police']       = df['quick_claim'] * df['no_police_report']
        df['repeat_claimer_quick']        = df['has_past_claims'] * df['quick_claim']
        df['urban_no_witness']            = df['urban_accident'] * df['no_witness']
        return df

    # ── STEP 4: TEMPORAL FEATURES ─────────────────────────────────────────

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        dow    = self._get(df, 'DayOfWeek', 'Monday')
        month  = self._get(df, 'Month', 'Jan')
        mclaim = self._get(df, 'MonthClaimed', 'Jan')
        dowc   = self._get(df, 'DayOfWeekClaimed', 'Monday')

        df['is_weekend']       = dow.isin(['Saturday', 'Sunday']).astype(int)
        df['is_end_of_year']   = month.isin(['Nov', 'Dec']).astype(int)
        df['claim_same_month'] = (month == mclaim).astype(int)

        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            '0': 0
        }
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6, '0': -1
        }

        df['Month_numeric']            = month.map(month_map).fillna(0)
        df['MonthClaimed_numeric']     = mclaim.map(month_map).fillna(0)
        df['DayOfWeek_numeric']        = dow.map(day_map).fillna(-1)
        df['DayOfWeekClaimed_numeric'] = dowc.map(day_map).fillna(-1)

        return df

    # ── STEP 5: BINARY CATEGORICALS ───────────────────────────────────────

    def _encode_binary_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        binary_mappings = {
            'PoliceReportFiled': {'No': 0, 'Yes': 1},
            'WitnessPresent':    {'No': 0, 'Yes': 1},
            'Fault':             {'Third Party': 0, 'Policy Holder': 1},
            'AgentType':         {'Internal': 0, 'External': 1},
            'AccidentArea':      {'Rural': 0, 'Urban': 1},
            'Sex':               {'Female': 0, 'Male': 1},
        }
        for col, mapping in binary_mappings.items():
            series = self._get(df, col, list(mapping.keys())[0])
            df[col + '_binary'] = series.map(mapping).fillna(0).astype(int)
        return df

    # ── STEP 6: ONE-HOT ENCODING ──────────────────────────────────────────

    def _one_hot_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Exact column names verified from feature_metadata.json
        known_dummies = {
            'Make': [
                'Make_BMW', 'Make_Chevrolet', 'Make_Dodge', 'Make_Ferrari',
                'Make_Ford', 'Make_Honda', 'Make_Jaguar', 'Make_Lexus',
                'Make_Mazda', 'Make_Mecedes',
                'Make_Mercury', 'Make_Nisson', 'Make_Pontiac', 'Make_Porche',
                'Make_Saab', 'Make_Saturn', 'Make_Toyota', 'Make_VW',
            ],
            'MaritalStatus': [
                'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Widow',
            ],
            'PolicyType': [
                'PolicyType_Sedan___Collision',
                'PolicyType_Sedan___Liability',
                'PolicyType_Sport___All_Perils',
                'PolicyType_Sport___Collision',
                'PolicyType_Sport___Liability',
                'PolicyType_Utility___All_Perils',
                'PolicyType_Utility___Collision',
                'PolicyType_Utility___Liability',
            ],
            'VehicleCategory': [
                'VehicleCategory_Sport',
                'VehicleCategory_Utility',
            ],
            'BasePolicy': [
                'BasePolicy_Collision',
                'BasePolicy_Liability',
            ],
        }

        for col in ['Make', 'MaritalStatus', 'PolicyType', 'VehicleCategory', 'BasePolicy']:
            series = self._get(df, col, 'Unknown')
            series = series.replace({'nan': 'Unknown', 'None': 'Unknown'})

            dummies = pd.get_dummies(series, prefix=col, drop_first=False)
            dummies.columns = (dummies.columns
                               .astype(str) 
                               .str.replace(' ', '_')
                               .str.replace('-', '_'))
            df = pd.concat([df, dummies], axis=1)

            for expected_col in known_dummies.get(col, []):
                if expected_col not in df.columns:
                    df[expected_col] = 0

        return df

    # ── STEP 7: SCALING ───────────────────────────────────────────────────

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        numerical_cols = ['Age', 'Deductible', 'DriverRating',
                          'WeekOfMonth', 'WeekOfMonthClaimed']
        ordinal_encoded_cols = [c for c in df.columns if c.endswith('_encoded')]
        numerical_cols = numerical_cols + ordinal_encoded_cols

        for col in numerical_cols:
            if col not in df.columns:
                df[col] = 0

        numerical_cols = [c for c in numerical_cols if c in df.columns]

        if not self._scaler_loaded or self.scaler is None:
            logger.warning("Scaler not loaded — skipping scaling.")
            return df

        try:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        except Exception as e:
            logger.error(f"Scaling failed: {e} — returning unscaled features")

        return df

    # ── STEP 8: SELECT FINAL FEATURES ────────────────────────────────────

    def _select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        drop_cols = [
            'AgeOfVehicle', 'VehiclePrice', 'AgeOfPolicyHolder',
            'PastNumberOfClaims', 'Days:Policy-Accident', 'Days:Policy-Claim',
            'NumberOfSuppliments', 'AddressChange-Claim', 'NumberOfCars',
            'PoliceReportFiled', 'WitnessPresent', 'Fault', 'AgentType',
            'AccidentArea', 'Sex', 'Make', 'MaritalStatus', 'PolicyType',
            'VehicleCategory', 'BasePolicy', 'Month', 'DayOfWeek',
            'MonthClaimed', 'DayOfWeekClaimed', 'FraudFound_P',
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        return df