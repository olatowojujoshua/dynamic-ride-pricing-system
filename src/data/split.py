"""
Data splitting utilities for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from ..config import TRAIN_TEST_SPLIT, RANDOM_SEED, TARGET_COLUMN
from ..utils.logger import logger
from ..utils.seed import set_random_seed

def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = TRAIN_TEST_SPLIT,
    random_state: int = RANDOM_SEED,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        stratify_column: Column to stratify by (optional)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data into train/test sets (test_size={test_size})")
    
    set_random_seed(random_state)
    
    # Determine stratification
    stratify = None
    if stratify_column and stratify_column in df.columns:
        stratify = df[stratify_column]
        logger.info(f"Stratifying by {stratify_column}")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    
    return train_df, test_df

def time_based_split(
    df: pd.DataFrame,
    time_column: str = 'Time_of_Booking',
    test_size: float = TRAIN_TEST_SPLIT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data based on time to simulate real-world deployment
    
    Args:
        df: DataFrame to split
        time_column: Column containing time information
        test_size: Proportion of data for testing (latest data)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Performing time-based split (test_size={test_size})")
    
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")
    
    # Sort by time column (assuming it can be ordered)
    df_sorted = df.copy()
    
    # For categorical time columns, we'll use a simple ordering
    if df_sorted[time_column].dtype == 'object':
        # Create a simple ordering for time categories
        time_order = ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night']
        df_sorted['time_order'] = df_sorted[time_column].map(
            lambda x: time_order.index(x) if x in time_order else 0
        )
        sort_column = 'time_order'
    else:
        sort_column = time_column
    
    df_sorted = df_sorted.sort_values(sort_column)
    
    # Calculate split point
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    # Remove temporary column if created
    if 'time_order' in train_df.columns:
        train_df = train_df.drop('time_order', axis=1)
        test_df = test_df.drop('time_order', axis=1)
    
    logger.info(f"Time-based split - Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    return train_df, test_df

def create_validation_split(
    train_df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create validation split from training data
    
    Args:
        train_df: Training DataFrame
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        stratify_column: Column to stratify by (optional)
    
    Returns:
        Tuple of (train_final_df, val_df)
    """
    logger.info(f"Creating validation split from training data (val_size={val_size})")
    
    set_random_seed(random_state)
    
    # Determine stratification
    stratify = None
    if stratify_column and stratify_column in train_df.columns:
        stratify = train_df[stratify_column]
    
    train_final_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"Final train: {len(train_final_df)} rows, Validation: {len(val_df)} rows")
    
    return train_final_df, val_df

def get_feature_target_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_columns: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        drop_columns: Additional columns to drop from features
    
    Returns:
        Tuple of (features_df, target_series)
    """
    if drop_columns is None:
        drop_columns = []
    
    # Ensure target column is included in drop columns
    if target_column not in drop_columns:
        drop_columns = drop_columns + [target_column]
    
    features = df.drop(columns=drop_columns, errors='ignore')
    target = df[target_column]
    
    logger.info(f"Features shape: {features.shape}, Target shape: {target.shape}")
    
    return features, target

def split_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Generate a report on the data split
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        val_df: Validation DataFrame (optional)
    
    Returns:
        Dictionary containing split statistics
    """
    report = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_percentage': len(train_df) / (len(train_df) + len(test_df)) * 100,
        'test_percentage': len(test_df) / (len(train_df) + len(test_df)) * 100,
        'total_size': len(train_df) + len(test_df)
    }
    
    if val_df is not None:
        report['val_size'] = len(val_df)
        report['val_percentage'] = len(val_df) / len(train_df) * 100
        report['total_with_val'] = len(train_df) + len(test_df) + len(val_df)
    
    # Target distribution
    if TARGET_COLUMN in train_df.columns:
        report['target_stats'] = {
            'train_mean': float(train_df[TARGET_COLUMN].mean()),
            'train_std': float(train_df[TARGET_COLUMN].std()),
            'test_mean': float(test_df[TARGET_COLUMN].mean()),
            'test_std': float(test_df[TARGET_COLUMN].std())
        }
        
        if val_df is not None and TARGET_COLUMN in val_df.columns:
            report['target_stats']['val_mean'] = float(val_df[TARGET_COLUMN].mean())
            report['target_stats']['val_std'] = float(val_df[TARGET_COLUMN].std())
    
    return report
