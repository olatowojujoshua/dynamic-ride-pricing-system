"""
Configuration settings for the Dynamic Ride Pricing System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data files
RAW_DATASET = RAW_DATA_DIR / "dynamic_pricing.csv"
PROCESSED_DATASET = PROCESSED_DATA_DIR / "processed_data.csv"

# Model files
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
SURGE_MODEL_PATH = MODELS_DIR / "surge_model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.json"

# Random seed for reproducibility
RANDOM_SEED = 42

# Feature columns
NUMERICAL_FEATURES = [
    'Number_of_Riders',
    'Number_of_Drivers', 
    'Number_of_Past_Rides',
    'Average_Ratings',
    'Expected_Ride_Duration'
]

CATEGORICAL_FEATURES = [
    'Location_Category',
    'Customer_Loyalty_Status',
    'Time_of_Booking',
    'Vehicle_Type'
]

TARGET_COLUMN = 'Historical_Cost_of_Ride'

# Pricing constraints
MIN_SURGE_MULTIPLIER = 0.8
MAX_SURGE_MULTIPLIER = 3.0
BASE_FLOOR_PRICE = 5.0
BASE_CEILING_PRICE = 100.0

# Fairness parameters
LOYALTY_DISCOUNTS = {
    'Silver': 0.05,
    'Gold': 0.10,
    'Platinum': 0.15
}

LOCATION_PREMIUMS = {
    'Urban': 1.0,
    'Suburban': 0.9,
    'Rural': 0.8
}

# Model parameters
TRAIN_TEST_SPLIT = 0.2
CV_FOLDS = 5

# Surge pricing thresholds
HIGH_DEMAND_THRESHOLD = 1.5  # riders/drivers ratio
LOW_SUPPLY_THRESHOLD = 0.5   # drivers/riders ratio

# Time-based pricing
RUSH_HOUR_START = 7
RUSH_HOUR_END = 19
WEEKEND_SURGE = 1.1

# Evaluation metrics
TARGET_MAPE = 0.15  # Maximum acceptable Mean Absolute Percentage Error
MIN_REVENUE_LIFT = 0.05  # Minimum 5% revenue lift expected
