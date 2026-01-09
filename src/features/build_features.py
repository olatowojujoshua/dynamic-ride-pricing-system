"""
Main feature engineering pipeline for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json

from ..config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, PROCESSED_DATASET
from ..utils.logger import logger
from .time_features import extract_time_features, create_time_buckets, create_time_interaction_features, calculate_time_based_multipliers
from .pressure_index import calculate_pressure_index, create_surge_indicators, create_location_pressure_features, calculate_elasticity_features, create_pressure_interaction_features

class FeatureBuilder:
    """
    Comprehensive feature engineering pipeline for dynamic pricing
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the feature engineering pipeline and transform the data
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed DataFrame with engineered features
        """
        logger.info("Fitting and transforming features")
        
        df_features = df.copy()
        
        # Step 1: Extract time features
        df_features = extract_time_features(df_features)
        df_features = create_time_buckets(df_features)
        df_features = create_time_interaction_features(df_features)
        df_features = calculate_time_based_multipliers(df_features)
        
        # Step 2: Calculate pressure index features
        df_features = calculate_pressure_index(df_features)
        df_features = create_surge_indicators(df_features)
        df_features = create_location_pressure_features(df_features)
        df_features = calculate_elasticity_features(df_features)
        df_features = create_pressure_interaction_features(df_features)
        
        # Step 3: Encode categorical variables
        df_features = self._encode_categorical_features(df_features, fit=True)
        
        # Step 4: Scale numerical features
        df_features = self._scale_numerical_features(df_features, fit=True)
        
        # Store feature columns
        self.feature_columns = [col for col in df_features.columns if col != TARGET_COLUMN]
        self.is_fitted = True
        
        logger.info(f"Feature engineering completed. Final shape: {df_features.shape}")
        
        return df_features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature engineering pipeline
        
        Args:
            df: Input DataFrame to transform
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("FeatureBuilder must be fitted before transforming data")
        
        logger.info("Transforming features using fitted pipeline")
        
        df_features = df.copy()
        
        # Apply the same feature engineering steps
        df_features = extract_time_features(df_features)
        df_features = create_time_buckets(df_features)
        df_features = create_time_interaction_features(df_features)
        df_features = calculate_time_based_multipliers(df_features)
        
        df_features = calculate_pressure_index(df_features)
        df_features = create_surge_indicators(df_features)
        df_features = create_location_pressure_features(df_features)
        df_features = calculate_elasticity_features(df_features)
        df_features = create_pressure_interaction_features(df_features)
        
        # Encode categorical variables using fitted encoders
        df_features = self._encode_categorical_features(df_features, fit=False)
        
        # Scale numerical features using fitted scalers
        df_features = self._scale_numerical_features(df_features, fit=False)
        
        return df_features
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders or use existing ones
        
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Original categorical features
        categorical_cols = CATEGORICAL_FEATURES
        
        # Additional categorical features created during engineering
        additional_categorical = ['time_period', 'demand_level', 'surge_level', 'time_location']
        categorical_cols.extend([col for col in additional_categorical if col in df_encoded.columns])
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    encoder = LabelEncoder()
                    # Handle unseen values by adding them to a special category
                    df_encoded[col] = df_encoded[col].fillna('Unknown')
                    df_encoded[col] = df_encoded[col].astype(str)
                    encoder.fit(df_encoded[col])
                    self.encoders[col] = encoder
                
                # Transform using fitted encoder
                if col in self.encoders:
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].fillna('Unknown').astype(str)
                    mask = df_encoded[col].isin(self.encoders[col].classes_)
                    df_encoded.loc[~mask, col] = 'Unknown'
                    
                    # Add 'Unknown' to classes if not present
                    if 'Unknown' not in self.encoders[col].classes_:
                        self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'Unknown')
                    
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def _scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scalers or use existing ones
        
        Returns:
            DataFrame with scaled numerical features
        """
        df_scaled = df.copy()
        
        # Identify numerical features (excluding target)
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET_COLUMN in numerical_cols:
            numerical_cols.remove(TARGET_COLUMN)
        
        for col in numerical_cols:
            if col in df_scaled.columns:
                if fit:
                    scaler = StandardScaler()
                    df_scaled[col] = scaler.fit_transform(df_scaled[[col]]).flatten()
                    self.scalers[col] = scaler
                else:
                    if col in self.scalers:
                        df_scaled[col] = self.scalers[col].transform(df_scaled[[col]]).flatten()
        
        return df_scaled
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by their importance category
        
        Returns:
            Dictionary mapping feature groups to feature names
        """
        if not self.is_fitted:
            raise ValueError("FeatureBuilder must be fitted first")
        
        feature_groups = {
            'time_features': [
                'booking_hour', 'hour_sin', 'hour_cos', 'is_rush_hour', 
                'is_weekend', 'demand_score', 'time_multiplier'
            ],
            'pressure_features': [
                'demand_supply_ratio', 'supply_demand_ratio', 'pressure_index',
                'market_imbalance', 'base_surge_multiplier', 'surge_probability'
            ],
            'location_features': [
                'Location_Category', 'location_pressure_multiplier', 'is_urban',
                'location_demand_interaction', 'location_supply_interaction'
            ],
            'customer_features': [
                'Customer_Loyalty_Status', 'Number_of_Past_Rides', 'Average_Ratings'
            ],
            'trip_features': [
                'Vehicle_Type', 'Expected_Ride_Duration'
            ],
            'interaction_features': [
                'time_location', 'rush_hour_urban', 'weekend_rush',
                'rush_hour_pressure', 'urban_pressure', 'loyalty_pressure_resistance'
            ]
        }
        
        # Filter to only include features that actually exist
        available_features = set(self.feature_columns)
        filtered_groups = {}
        
        for group, features in feature_groups.items():
            filtered_groups[group] = [f for f in features if f in available_features]
        
        return filtered_groups
    
    def save_encoders(self, filepath: str) -> None:
        """
        Save encoders and scalers to file
        
        Args:
            filepath: Path to save the encoders
        """
        encoder_data = {
            'encoders': {k: v.classes_.tolist() for k, v in self.encoders.items()},
            'scalers': {k: {'mean': v.mean_.tolist(), 'scale': v.scale_.tolist()} 
                       for k, v in self.scalers.items()},
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'w') as f:
            json.dump(encoder_data, f, indent=2)
        
        logger.info(f"Saved encoders and scalers to {filepath}")
    
    def load_encoders(self, filepath: str) -> None:
        """
        Load encoders and scalers from file
        
        Args:
            filepath: Path to load the encoders from
        """
        with open(filepath, 'r') as f:
            encoder_data = json.load(f)
        
        # Reconstruct encoders
        self.encoders = {}
        for col, classes in encoder_data['encoders'].items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(classes)
            self.encoders[col] = encoder
        
        # Reconstruct scalers
        self.scalers = {}
        for col, scaler_data in encoder_data['scalers'].items():
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_data['mean'])
            scaler.scale_ = np.array(scaler_data['scale'])
            self.scalers[col] = scaler
        
        self.feature_columns = encoder_data['feature_columns']
        self.is_fitted = True
        
        logger.info(f"Loaded encoders and scalers from {filepath}")

def build_features(df: pd.DataFrame, fit_transform: bool = True) -> Tuple[pd.DataFrame, FeatureBuilder]:
    """
    Build features for the dynamic pricing model
    
    Args:
        df: Input DataFrame
        fit_transform: Whether to fit the feature pipeline or just transform
    
    Returns:
        Tuple of (engineered DataFrame, FeatureBuilder instance)
    """
    feature_builder = FeatureBuilder()
    
    if fit_transform:
        df_features = feature_builder.fit_transform(df)
    else:
        df_features = feature_builder.transform(df)
    
    return df_features, feature_builder

def create_feature_report(df: pd.DataFrame, feature_builder: FeatureBuilder) -> Dict[str, Any]:
    """
    Create a comprehensive feature engineering report
    
    Args:
        df: Engineered DataFrame
        feature_builder: Fitted FeatureBuilder instance
    
    Returns:
        Dictionary containing feature report
    """
    report = {
        'total_features': len(df.columns),
        'feature_groups': feature_builder.get_feature_importance_groups(),
        'feature_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'feature_statistics': {}
    }
    
    # Statistics for numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    for col in numerical_features:
        report['feature_statistics'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'quartiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict()
        }
    
    return report
