"""
Data cleaning utilities for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import stats
from ..config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
from ..utils.logger import logger

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform comprehensive data cleaning
    
    Args:
        df: Raw DataFrame to clean
    
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    logger.info("Starting data cleaning process")
    
    cleaning_report = {
        'original_shape': df.shape,
        'cleaning_steps': [],
        'final_shape': None,
        'removed_rows': 0,
        'modified_columns': []
    }
    
    df_clean = df.copy()
    
    # Step 1: Handle missing values
    df_clean, missing_report = handle_missing_values(df_clean)
    cleaning_report['cleaning_steps'].append(f"Missing values handled: {missing_report}")
    cleaning_report['removed_rows'] += missing_report.get('removed_rows', 0)
    
    # Step 2: Detect and handle outliers
    df_clean, outlier_report = detect_outliers(df_clean)
    cleaning_report['cleaning_steps'].append(f"Outliers handled: {outlier_report}")
    cleaning_report['removed_rows'] += outlier_report.get('removed_rows', 0)
    
    # Step 3: Validate data ranges
    df_clean, range_report = validate_data_ranges(df_clean)
    cleaning_report['cleaning_steps'].append(f"Data ranges validated: {range_report}")
    
    # Step 4: Standardize categorical values
    df_clean, cat_report = standardize_categorical(df_clean)
    cleaning_report['cleaning_steps'].append(f"Categorical data standardized: {cat_report}")
    if cat_report.get('modified_columns'):
        cleaning_report['modified_columns'].extend(cat_report['modified_columns'])
    
    cleaning_report['final_shape'] = df_clean.shape
    
    logger.info(f"Data cleaning completed. Removed {cleaning_report['removed_rows']} rows")
    logger.info(f"Final shape: {cleaning_report['final_shape']}")
    
    return df_clean, cleaning_report

def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values in the dataset
    
    Args:
        df: DataFrame with missing values
    
    Returns:
        Tuple of (cleaned DataFrame, missing values report)
    """
    logger.info("Handling missing values")
    
    df_clean = df.copy()
    missing_report = {
        'missing_before': df.isnull().sum().to_dict(),
        'missing_after': None,
        'removed_rows': 0,
        'imputed_columns': []
    }
    
    # Check missing values by column
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    
    # Remove rows with missing target values
    if TARGET_COLUMN in df.columns:
        target_missing = df[TARGET_COLUMN].isnull().sum()
        if target_missing > 0:
            df_clean = df_clean.dropna(subset=[TARGET_COLUMN])
            missing_report['removed_rows'] = target_missing
            logger.info(f"Removed {target_missing} rows with missing target values")
    
    # Handle numerical features
    for col in NUMERICAL_FEATURES:
        if col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                # Use median for numerical features (robust to outliers)
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)
                missing_report['imputed_columns'].append(f"{col} (median: {median_value:.2f})")
                logger.info(f"Imputed {col} with median value: {median_value:.2f}")
    
    # Handle categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                # Use mode for categorical features
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
                missing_report['imputed_columns'].append(f"{col} (mode: {mode_value})")
                logger.info(f"Imputed {col} with mode value: {mode_value}")
    
    missing_report['missing_after'] = df_clean.isnull().sum().to_dict()
    
    return df_clean, missing_report

def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and handle outliers in numerical features
    
    Args:
        df: DataFrame to process
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Tuple of (cleaned DataFrame, outlier report)
    """
    logger.info(f"Detecting outliers using {method} method")
    
    df_clean = df.copy()
    outlier_report = {
        'method': method,
        'threshold': threshold,
        'outliers_detected': {},
        'removed_rows': 0
    }
    
    total_outliers = 0
    
    for col in NUMERICAL_FEATURES:
        if col in df_clean.columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outliers = z_scores > threshold
                # Align the boolean array with the DataFrame
                outliers = pd.Series(outliers, index=df_clean[col].dropna().index).reindex(df_clean.index, fill_value=False)
            
            col_outliers = outliers.sum()
            if col_outliers > 0:
                outlier_report['outliers_detected'][col] = {
                    'count': int(col_outliers),
                    'percentage': float(col_outliers / len(df_clean) * 100)
                }
                total_outliers += col_outliers
                
                # Cap outliers instead of removing them (less aggressive)
                if method == 'iqr':
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                elif method == 'zscore':
                    median_val = df_clean[col].median()
                    df_clean.loc[outliers, col] = median_val
                
                logger.info(f"Capped {col_outliers} outliers in {col}")
    
    outlier_report['total_outliers'] = total_outliers
    
    return df_clean, outlier_report

def validate_data_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and correct data ranges for logical consistency
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (validated DataFrame, validation report)
    """
    logger.info("Validating data ranges")
    
    df_clean = df.copy()
    validation_report = {
        'validations': [],
        'corrected_rows': 0
    }
    
    # Validate Number_of_Riders and Number_of_Drivers (should be positive)
    for col in ['Number_of_Riders', 'Number_of_Drivers']:
        if col in df_clean.columns:
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean.loc[df_clean[col] < 0, col] = 0
                validation_report['validations'].append(f"Set {negative_count} negative values in {col} to 0")
                validation_report['corrected_rows'] += negative_count
    
    # Validate Expected_Ride_Duration (should be positive and reasonable)
    if 'Expected_Ride_Duration' in df_clean.columns:
        invalid_duration = (df_clean['Expected_Ride_Duration'] <= 0) | (df_clean['Expected_Ride_Duration'] > 300)  # Max 5 hours
        invalid_count = invalid_duration.sum()
        if invalid_count > 0:
            median_duration = df_clean['Expected_Ride_Duration'].median()
            df_clean.loc[invalid_duration, 'Expected_Ride_Duration'] = median_duration
            validation_report['validations'].append(f"Corrected {invalid_count} invalid ride durations")
            validation_report['corrected_rows'] += invalid_count
    
    # Validate Average_Ratings (should be between 1 and 5)
    if 'Average_Ratings' in df_clean.columns:
        invalid_ratings = (df_clean['Average_Ratings'] < 1) | (df_clean['Average_Ratings'] > 5)
        invalid_count = invalid_ratings.sum()
        if invalid_count > 0:
            median_rating = df_clean['Average_Ratings'].median()
            df_clean.loc[invalid_ratings, 'Average_Ratings'] = median_rating
            validation_report['validations'].append(f"Corrected {invalid_count} invalid ratings")
            validation_report['corrected_rows'] += invalid_count
    
    # Validate Historical_Cost_of_Ride (should be positive)
    if TARGET_COLUMN in df_clean.columns:
        negative_cost = (df_clean[TARGET_COLUMN] <= 0).sum()
        if negative_cost > 0:
            median_cost = df_clean[TARGET_COLUMN].median()
            df_clean.loc[df_clean[TARGET_COLUMN] <= 0, TARGET_COLUMN] = median_cost
            validation_report['validations'].append(f"Corrected {negative_cost} non-positive costs")
            validation_report['corrected_rows'] += negative_count
    
    return df_clean, validation_report

def standardize_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardize categorical values (remove extra spaces, capitalize properly)
    
    Args:
        df: DataFrame to standardize
    
    Returns:
        Tuple of (standardized DataFrame, standardization report)
    """
    logger.info("Standardizing categorical values")
    
    df_clean = df.copy()
    standardization_report = {
        'modified_columns': [],
        'changes': {}
    }
    
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Remove leading/trailing whitespace and convert to title case
                original_values = df_clean[col].unique().tolist()
                df_clean[col] = df_clean[col].str.strip().str.title()
                new_values = df_clean[col].unique().tolist()
                
                if original_values != new_values:
                    standardization_report['modified_columns'].append(col)
                    standardization_report['changes'][col] = {
                        'original': original_values,
                        'new': new_values
                    }
                    logger.info(f"Standardized values in {col}")
    
    return df_clean, standardization_report
