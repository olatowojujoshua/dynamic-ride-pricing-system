"""
Time-based feature engineering for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..config import RUSH_HOUR_START, RUSH_HOUR_END, WEEKEND_SURGE
from ..utils.logger import logger

def extract_time_features(df: pd.DataFrame, time_column: str = 'Time_of_Booking') -> pd.DataFrame:
    """
    Extract time-based features from booking time
    
    Args:
        df: Input DataFrame
        time_column: Name of the time column
    
    Returns:
        DataFrame with additional time features
    """
    logger.info("Extracting time-based features")
    
    df_features = df.copy()
    
    if time_column not in df.columns:
        logger.warning(f"Time column '{time_column}' not found. Skipping time feature extraction.")
        return df_features
    
    # Create hour-based features (assuming categorical time data)
    if df_features[time_column].dtype == 'object':
        # Map time categories to hour ranges
        time_to_hour = {
            'Early Morning': 5,
            'Morning': 8,
            'Afternoon': 14,
            'Evening': 18,
            'Night': 22,
            'Late Night': 1
        }
        
        df_features['booking_hour'] = df_features[time_column].map(time_to_hour)
        df_features['booking_hour'] = df_features['booking_hour'].fillna(12)  # Default to noon
        
        # Create cyclical features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['booking_hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['booking_hour'] / 24)
        
        # Rush hour indicator
        df_features['is_rush_hour'] = ((df_features['booking_hour'] >= RUSH_HOUR_START) & 
                                      (df_features['booking_hour'] <= RUSH_HOUR_END)).astype(int)
        
        # Time of day categories
        df_features['time_period'] = pd.cut(
            df_features['booking_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        ).astype('object')
        
        # Weekend indicator (simplified - assume uniform distribution)
        # In real data, you'd extract this from actual timestamps
        np.random.seed(42)
        df_features['is_weekend'] = np.random.choice([0, 1], size=len(df_features), p=[0.7, 0.3])
        
        logger.info("Created time-based features from categorical time data")
    
    return df_features

def create_time_buckets(df: pd.DataFrame, time_column: str = 'Time_of_Booking') -> pd.DataFrame:
    """
    Create time-based buckets for pricing analysis
    
    Args:
        df: Input DataFrame
        time_column: Name of the time column
    
    Returns:
        DataFrame with time bucket features
    """
    logger.info("Creating time buckets")
    
    df_buckets = df.copy()
    
    if time_column not in df.columns:
        logger.warning(f"Time column '{time_column}' not found. Skipping time bucket creation.")
        return df_buckets
    
    # Create demand level buckets based on time
    if df_buckets[time_column].dtype == 'object':
        # Map time categories to demand levels
        demand_mapping = {
            'Early Morning': 'Low',
            'Morning': 'High', 
            'Afternoon': 'Medium',
            'Evening': 'High',
            'Night': 'Medium',
            'Late Night': 'Low'
        }
        
        df_buckets['demand_level'] = df_buckets[time_column].map(demand_mapping)
        
        # Create numeric demand score
        demand_score_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df_buckets['demand_score'] = df_buckets['demand_level'].map(demand_score_mapping)
        
        logger.info("Created demand level buckets")
    
    return df_buckets

def is_rush_hour(hour: int) -> bool:
    """
    Check if a given hour is during rush hour
    
    Args:
        hour: Hour of the day (0-23)
    
    Returns:
        True if it's rush hour, False otherwise
    """
    return RUSH_HOUR_START <= hour <= RUSH_HOUR_END

def create_time_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between time and other variables
    
    Args:
        df: Input DataFrame with time features
    
    Returns:
        DataFrame with time interaction features
    """
    logger.info("Creating time interaction features")
    
    df_interactions = df.copy()
    
    # Time x Location interactions
    if 'time_period' in df_interactions.columns and 'Location_Category' in df_interactions.columns:
        df_interactions['time_location'] = df_interactions['time_period'] + '_' + df_interactions['Location_Category']
    
    # Rush hour x Location interactions
    if 'is_rush_hour' in df_interactions.columns and 'Location_Category' in df_interactions.columns:
        df_interactions['rush_hour_urban'] = (
            df_interactions['is_rush_hour'] * 
            (df_interactions['Location_Category'] == 'Urban').astype(int)
        )
    
    # Weekend x Time interactions
    if 'is_weekend' in df_interactions.columns and 'is_rush_hour' in df_interactions.columns:
        df_interactions['weekend_rush'] = df_interactions['is_weekend'] * df_interactions['is_rush_hour']
    
    logger.info("Created time interaction features")
    
    return df_interactions

def calculate_time_based_multipliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based pricing multipliers
    
    Args:
        df: Input DataFrame with time features
    
    Returns:
        DataFrame with time-based multiplier columns
    """
    logger.info("Calculating time-based pricing multipliers")
    
    df_multipliers = df.copy()
    
    # Base time multiplier
    df_multipliers['time_multiplier'] = 1.0
    
    # Rush hour multiplier
    if 'is_rush_hour' in df_multipliers.columns:
        df_multipliers.loc[df_multipliers['is_rush_hour'] == 1, 'time_multiplier'] *= 1.2
    
    # Weekend multiplier
    if 'is_weekend' in df_multipliers.columns:
        df_multipliers.loc[df_multipliers['is_weekend'] == 1, 'time_multiplier'] *= WEEKEND_SURGE
    
    # Late night premium
    if 'booking_hour' in df_multipliers.columns:
        late_night = (df_multipliers['booking_hour'] >= 22) | (df_multipliers['booking_hour'] <= 5)
        df_multipliers.loc[late_night, 'time_multiplier'] *= 1.15
    
    logger.info("Calculated time-based pricing multipliers")
    
    return df_multipliers

def get_time_feature_importance(df: pd.DataFrame, target_column: str = 'Historical_Cost_of_Ride') -> Dict[str, float]:
    """
    Calculate basic feature importance for time-based features
    
    Args:
        df: DataFrame with time features and target
        target_column: Name of target column
    
    Returns:
        Dictionary of feature importance scores
    """
    time_features = [
        'booking_hour', 'hour_sin', 'hour_cos', 'is_rush_hour', 
        'is_weekend', 'demand_score', 'time_multiplier'
    ]
    
    importance_scores = {}
    
    for feature in time_features:
        if feature in df.columns and target_column in df.columns:
            correlation = df[feature].corr(df[target_column])
            importance_scores[feature] = abs(correlation) if not pd.isna(correlation) else 0.0
    
    return importance_scores
