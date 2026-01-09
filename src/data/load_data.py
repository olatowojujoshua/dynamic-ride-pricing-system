"""
Data loading utilities for the Dynamic Ride Pricing System
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from ..config import RAW_DATASET, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
from ..utils.logger import logger

def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw dataset from CSV file
    
    Args:
        file_path: Path to the raw dataset file. If None, uses default from config.
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if file_path is None:
        file_path = RAW_DATASET
    
    logger.info(f"Loading raw data from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file {file_path} is empty")
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {str(e)}")

def validate_data_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the dataset schema and return validation results
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary containing validation results
    """
    logger.info("Validating data schema")
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    # Check required columns
    required_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    expected_types = {}
    for col in NUMERICAL_FEATURES + [TARGET_COLUMN]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_results['warnings'].append(f"Column {col} should be numeric but is {df[col].dtype}")
            expected_types[col] = 'numeric'
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                validation_results['warnings'].append(f"Column {col} should be categorical but is {df[col].dtype}")
            expected_types[col] = 'categorical'
    
    # Basic statistics
    validation_results['info'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'expected_types': expected_types
    }
    
    if validation_results['is_valid']:
        logger.info("Data schema validation passed")
    else:
        logger.error("Data schema validation failed")
    
    return validation_results

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the dataset
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        Dictionary containing data summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numerical_summary': {},
        'categorical_summary': {}
    }
    
    # Numerical features summary
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            summary['numerical_summary'][col] = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'quartiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict()
            }
    
    # Categorical features summary
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            summary['categorical_summary'][col] = {
                'count': df[col].count(),
                'unique_count': df[col].nunique(),
                'unique_values': df[col].unique().tolist(),
                'value_counts': df[col].value_counts().to_dict()
            }
    
    return summary
