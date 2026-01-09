"""
Demand-supply pressure index calculation for dynamic pricing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from ..config import HIGH_DEMAND_THRESHOLD, LOW_SUPPLY_THRESHOLD
from ..utils.logger import logger

def calculate_pressure_index(
    df: pd.DataFrame,
    riders_column: str = 'Number_of_Riders',
    drivers_column: str = 'Number_of_Drivers'
) -> pd.DataFrame:
    """
    Calculate demand-supply pressure index and related features
    
    Args:
        df: Input DataFrame
        riders_column: Name of riders column
        drivers_column: Name of drivers column
    
    Returns:
        DataFrame with pressure index features
    """
    logger.info("Calculating demand-supply pressure index")
    
    df_pressure = df.copy()
    
    # Validate required columns
    if riders_column not in df_pressure.columns:
        raise ValueError(f"Riders column '{riders_column}' not found")
    if drivers_column not in df_pressure.columns:
        raise ValueError(f"Drivers column '{drivers_column}' not found")
    
    # Basic demand-supply ratio
    df_pressure['demand_supply_ratio'] = df_pressure[riders_column] / (df_pressure[drivers_column] + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Supply-demand ratio (inverse)
    df_pressure['supply_demand_ratio'] = df_pressure[drivers_column] / (df_pressure[riders_column] + 1e-8)
    
    # Absolute demand and supply levels
    df_pressure['demand_level'] = df_pressure[riders_column]
    df_pressure['supply_level'] = df_pressure[drivers_column]
    
    # Market pressure index (normalized)
    df_pressure['pressure_index'] = np.log1p(df_pressure['demand_supply_ratio'])
    
    # Scarcity indicators
    df_pressure['driver_scarcity'] = (df_pressure[drivers_column] < 5).astype(int)
    df_pressure['high_demand'] = (df_pressure[riders_column] > 20).astype(int)
    
    # Market imbalance score
    df_pressure['market_imbalance'] = np.abs(df_pressure['demand_supply_ratio'] - 1.0)
    
    # Competitive intensity
    df_pressure['competitive_intensity'] = df_pressure[riders_column] / (df_pressure[drivers_column] + 1)
    
    logger.info("Calculated pressure index features")
    
    return df_pressure

def create_surge_indicators(
    df: pd.DataFrame,
    ratio_column: str = 'demand_supply_ratio'
) -> pd.DataFrame:
    """
    Create surge pricing indicators based on demand-supply dynamics
    
    Args:
        df: Input DataFrame with pressure index
        ratio_column: Name of demand-supply ratio column
    
    Returns:
        DataFrame with surge indicator features
    """
    logger.info("Creating surge pricing indicators")
    
    df_surge = df.copy()
    
    if ratio_column not in df_surge.columns:
        raise ValueError(f"Ratio column '{ratio_column}' not found")
    
    # Surge levels
    conditions = [
        df_surge[ratio_column] >= 2.0,
        df_surge[ratio_column] >= 1.5,
        df_surge[ratio_column] >= 1.0,
        df_surge[ratio_column] >= 0.5,
        df_surge[ratio_column] < 0.5
    ]
    
    choices = ['Extreme', 'High', 'Moderate', 'Low', 'Very Low']
    
    df_surge['surge_level'] = np.select(conditions, choices, default='Moderate')
    
    # Numeric surge multiplier (base calculation)
    df_surge['base_surge_multiplier'] = np.clip(
        1.0 + (df_surge[ratio_column] - 1.0) * 0.3,  # 30% of excess demand translates to surge
        0.8,  # Minimum 80% of base price
        3.0   # Maximum 3x surge
    )
    
    # Surge triggers
    df_surge['high_demand_trigger'] = (df_surge[ratio_column] >= HIGH_DEMAND_THRESHOLD).astype(int)
    df_surge['low_supply_trigger'] = (df_surge[ratio_column] >= (1/LOW_SUPPLY_THRESHOLD)).astype(int)
    
    # Surge probability (simplified)
    df_surge['surge_probability'] = np.clip(
        (df_surge[ratio_column] - 0.5) / 2.0,  # Normalize to 0-1 range
        0.0,
        1.0
    )
    
    # Emergency surge (extreme imbalance)
    df_surge['emergency_surge'] = (df_surge[ratio_column] >= 3.0).astype(int)
    
    logger.info("Created surge pricing indicators")
    
    return df_surge

def create_location_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create location-specific pressure features
    
    Args:
        df: Input DataFrame with location and pressure data
    
    Returns:
        DataFrame with location pressure features
    """
    logger.info("Creating location pressure features")
    
    df_location = df.copy()
    
    if 'Location_Category' not in df_location.columns:
        logger.warning("Location_Category not found. Skipping location pressure features.")
        return df_location
    
    # Location-based pressure multipliers
    location_pressure = {
        'Urban': 1.2,    # Higher base pressure in urban areas
        'Suburban': 1.0,
        'Rural': 0.8     # Lower pressure in rural areas
    }
    
    df_location['location_pressure_multiplier'] = df_location['Location_Category'].map(location_pressure)
    
    # Urban density indicator
    df_location['is_urban'] = (df_location['Location_Category'] == 'Urban').astype(int)
    
    # Location x Demand interaction
    if 'demand_level' in df_location.columns:
        df_location['location_demand_interaction'] = (
            df_location['is_urban'] * df_location['demand_level']
        )
    
    # Location x Supply interaction
    if 'supply_level' in df_location.columns:
        df_location['location_supply_interaction'] = (
            df_location['is_urban'] * df_location['supply_level']
        )
    
    logger.info("Created location pressure features")
    
    return df_location

def calculate_elasticity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price elasticity and related features
    
    Args:
        df: Input DataFrame with pricing and demand data
    
    Returns:
        DataFrame with elasticity features
    """
    logger.info("Calculating elasticity features")
    
    df_elasticity = df.copy()
    
    # Price sensitivity by location
    if 'Location_Category' in df_elasticity.columns and 'Historical_Cost_of_Ride' in df_elasticity.columns:
        location_avg_price = df_elasticity.groupby('Location_Category')['Historical_Cost_of_Ride'].transform('mean')
        df_elasticity['price_vs_location_avg'] = df_elasticity['Historical_Cost_of_Ride'] / location_avg_price
    
    # Demand elasticity indicators
    if 'demand_supply_ratio' in df_elasticity.columns:
        # Higher ratios should correlate with higher prices
        df_elasticity['demand_elasticity_factor'] = np.log1p(df_elasticity['demand_supply_ratio'])
    
    # Supply elasticity indicators
    if 'supply_demand_ratio' in df_elasticity.columns:
        # Lower supply should increase prices
        df_elasticity['supply_elasticity_factor'] = 1.0 / (df_elasticity['supply_demand_ratio'] + 1e-8)
    
    # Market efficiency score
    if all(col in df_elasticity.columns for col in ['demand_level', 'supply_level']):
        df_elasticity['market_efficiency'] = df_elasticity['supply_level'] / (df_elasticity['demand_level'] + 1e-8)
    
    logger.info("Calculated elasticity features")
    
    return df_elasticity

def create_pressure_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between pressure and other variables
    
    Args:
        df: Input DataFrame with pressure features
    
    Returns:
        DataFrame with pressure interaction features
    """
    logger.info("Creating pressure interaction features")
    
    df_interactions = df.copy()
    
    # Pressure x Time interactions
    if 'demand_supply_ratio' in df_interactions.columns and 'is_rush_hour' in df_interactions.columns:
        df_interactions['rush_hour_pressure'] = (
            df_interactions['is_rush_hour'] * df_interactions['demand_supply_ratio']
        )
    
    # Pressure x Location interactions
    if 'demand_supply_ratio' in df_interactions.columns and 'Location_Category' in df_interactions.columns:
        df_interactions['urban_pressure'] = (
            (df_interactions['Location_Category'] == 'Urban').astype(int) * 
            df_interactions['demand_supply_ratio']
        )
    
    # Pressure x Loyalty interactions
    if 'demand_supply_ratio' in df_interactions.columns and 'Customer_Loyalty_Status' in df_interactions.columns:
        # High-tier customers might be less sensitive to surge
        loyalty_resistance = {
            'Silver': 0.9,
            'Gold': 0.8,
            'Platinum': 0.7
        }
        df_interactions['loyalty_pressure_resistance'] = (
            df_interactions['Customer_Loyalty_Status'].map(loyalty_resistance).fillna(1.0) *
            df_interactions['demand_supply_ratio']
        )
    
    logger.info("Created pressure interaction features")
    
    return df_interactions

def get_pressure_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for pressure features
    
    Args:
        df: DataFrame with pressure features
    
    Returns:
        Dictionary containing pressure feature summaries
    """
    pressure_features = [
        'demand_supply_ratio', 'supply_demand_ratio', 'pressure_index',
        'market_imbalance', 'base_surge_multiplier', 'surge_probability'
    ]
    
    summary = {}
    
    for feature in pressure_features:
        if feature in df.columns:
            summary[feature] = {
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std()),
                'min': float(df[feature].min()),
                'max': float(df[feature].max()),
                'quartiles': df[feature].quantile([0.25, 0.5, 0.75]).to_dict()
            }
    
    return summary
