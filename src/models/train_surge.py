"""
Surge pricing model training for demand-supply dynamics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import xgboost as xgb
import lightgbm as lgb

from ..config import SURGE_MODEL_PATH, TARGET_COLUMN, CV_FOLDS, RANDOM_SEED
from ..utils.logger import logger
from ..utils.seed import set_random_seed

class SurgeModel:
    """
    Surge pricing model that predicts multipliers based on demand-supply dynamics
    """
    
    def __init__(self, model_type: str = 'random_forest', quantile_alpha: float = 0.9):
        """
        Initialize the surge model
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'quantile')
            quantile_alpha: Quantile for quantile regression (0.5 for median, 0.9 for upper bound)
        """
        self.model_type = model_type
        self.quantile_alpha = quantile_alpha
        self.model = None
        self.feature_importance = None
        self.is_fitted = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specific model type"""
        set_random_seed(RANDOM_SEED)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                loss='quantile',
                alpha=self.quantile_alpha,
                random_state=RANDOM_SEED
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                objective='reg:quantileerror',
                quantile_alpha=self.quantile_alpha
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                objective='quantile',
                alpha=self.quantile_alpha
            )
        elif self.model_type == 'quantile':
            self.model = QuantileRegressor(
                alpha=self.quantile_alpha,
                solver='highs'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} surge model with quantile alpha={self.quantile_alpha}")
    
    def _create_surge_target(self, df: pd.DataFrame, base_price_col: str = 'base_price') -> pd.Series:
        """
        Create surge multiplier target from historical data
        
        Args:
            df: DataFrame with price data
            base_price_col: Column name for base price
        
        Returns:
            Series with surge multipliers
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
        
        if base_price_col not in df.columns:
            # Estimate base price using simple features
            base_price = self._estimate_base_price(df)
        else:
            base_price = df[base_price_col]
        
        # Calculate surge multiplier (with bounds)
        surge_multiplier = df[TARGET_COLUMN] / base_price
        surge_multiplier = np.clip(surge_multiplier, 0.5, 3.0)  # Reasonable bounds
        
        return surge_multiplier
    
    def _estimate_base_price(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate base price using fundamental trip characteristics
        
        Args:
            df: DataFrame with trip data
        
        Returns:
            Series with estimated base prices
        """
        # Simple base price estimation based on duration and vehicle type
        base_price = df['Expected_Ride_Duration'] * 0.5  # $0.5 per minute
        
        # Vehicle type adjustments
        if 'Vehicle_Type' in df.columns:
            vehicle_multipliers = {
                'Economy': 1.0,
                'Premium': 1.5,
                'Luxury': 2.0
            }
            for vehicle_type, multiplier in vehicle_multipliers.items():
                mask = df['Vehicle_Type'] == vehicle_type
                base_price.loc[mask] *= multiplier
        
        # Location adjustments
        if 'Location_Category' in df.columns:
            location_multipliers = {
                'Urban': 1.2,
                'Suburban': 1.0,
                'Rural': 0.8
            }
            for location, multiplier in location_multipliers.items():
                mask = df['Location_Category'] == location
                base_price.loc[mask] *= multiplier
        
        return base_price
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Fit the surge model
        
        Args:
            X_train: Training features
            y_train: Training target (surge multipliers)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        logger.info(f"Training {self.model_type} surge model")
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = dict(zip(X_train.columns, np.abs(self.model.coef_)))
        
        # Log training metrics
        train_pred = self.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_pinball = mean_pinball_loss(y_train, train_pred, alpha=self.quantile_alpha)
        
        logger.info(f"Training metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, Pinball: {train_pinball:.4f}")
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_pinball = mean_pinball_loss(y_val, val_pred, alpha=self.quantile_alpha)
            
            logger.info(f"Validation metrics - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, Pinball: {val_pinball:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make surge multiplier predictions
        
        Args:
            X: Features to predict on
        
        Returns:
            Predicted surge multipliers
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        
        # Apply bounds to surge multipliers
        predictions = np.clip(predictions, 0.5, 3.0)
        
        return predictions
    
    def predict_interval(self, X: pd.DataFrame, lower_alpha: float = 0.1, upper_alpha: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict surge multipliers with confidence intervals
        
        Args:
            X: Features to predict on
            lower_alpha: Lower quantile alpha
            upper_alpha: Upper quantile alpha
        
        Returns:
            Tuple of (median, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Train temporary models for different quantiles if needed
        if self.model_type in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            # Use the same model for all quantiles (approximation)
            median_pred = self.predict(X)
            
            # Create bounds based on training data variance
            if hasattr(self.model, 'estimators_') and self.model_type == 'random_forest':
                # Use individual tree predictions for uncertainty
                tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
                lower_bound = np.percentile(tree_preds, lower_alpha * 100, axis=0)
                upper_bound = np.percentile(tree_preds, upper_alpha * 100, axis=0)
            else:
                # Simple uncertainty estimation
                std_estimate = 0.2 * median_pred  # Assume 20% relative uncertainty
                lower_bound = np.clip(median_pred - 1.96 * std_estimate, 0.5, 3.0)
                upper_bound = np.clip(median_pred + 1.96 * std_estimate, 0.5, 3.0)
        else:
            # For quantile models, train separate models for each quantile
            lower_model = SurgeModel(self.model_type, lower_alpha)
            upper_model = SurgeModel(self.model_type, upper_alpha)
            
            # Note: In practice, you'd want to train these models properly
            # This is a simplified approach
            median_pred = self.predict(X)
            lower_bound = np.clip(median_pred * 0.8, 0.5, 3.0)
            upper_bound = np.clip(median_pred * 1.2, 0.5, 3.0)
        
        return median_pred, lower_bound, upper_bound
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = CV_FOLDS) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target (surge multipliers)
            cv: Number of CV folds
        
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Negative MSE because sklearn's cross_val_score uses negative values for loss functions
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)
        
        # Pinball loss for quantile regression
        pinball_scores = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            temp_model = SurgeModel(self.model_type, self.quantile_alpha)
            temp_model.fit(X_train_fold, y_train_fold)
            pred = temp_model.predict(X_val_fold)
            pinball = mean_pinball_loss(y_val_fold, pred, alpha=self.quantile_alpha)
            pinball_scores.append(pinball)
        
        cv_results = {
            'mean_rmse': cv_rmse.mean(),
            'std_rmse': cv_rmse.std(),
            'rmse_scores': cv_rmse.tolist(),
            'mean_pinball': np.mean(pinball_scores),
            'std_pinball': np.std(pinball_scores),
            'pinball_scores': pinball_scores
        }
        
        logger.info(f"CV RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
        logger.info(f"CV Pinball: {cv_results['mean_pinball']:.4f} ± {cv_results['std_pinball']:.4f}")
        
        return cv_results
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top feature importance
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            Dictionary of top features and their importance scores
        """
        if not self.feature_importance:
            return {}
        
        # Sort by importance and return top N
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)
    
    def save_model(self, filepath: str = SURGE_MODEL_PATH):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'quantile_alpha': self.quantile_alpha,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved surge model to {filepath}")
    
    def load_model(self, filepath: str = SURGE_MODEL_PATH):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.quantile_alpha = model_data['quantile_alpha']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Loaded surge model from {filepath}")

def train_surge_model(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    model_type: str = 'random_forest',
    quantile_alpha: float = 0.9
) -> Tuple[SurgeModel, Dict[str, Any]]:
    """
    Train a surge pricing model
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame (optional)
        feature_columns: List of feature columns to use
        model_type: Type of model to train
        quantile_alpha: Quantile for quantile regression
    
    Returns:
        Tuple of (trained model, training report)
    """
    logger.info(f"Training surge model: {model_type}")
    
    # Initialize model
    model = SurgeModel(model_type=model_type, quantile_alpha=quantile_alpha)
    
    # Create surge target
    y_train = model._create_surge_target(df_train)
    
    # Select features
    if feature_columns is None:
        # Use pressure-related features by default
        pressure_features = [
            'demand_supply_ratio', 'supply_demand_ratio', 'pressure_index',
            'market_imbalance', 'base_surge_multiplier', 'surge_probability',
            'is_rush_hour', 'is_weekend', 'Location_Category', 'time_multiplier'
        ]
        feature_columns = [col for col in pressure_features if col in df_train.columns]
    
    X_train = df_train[feature_columns]
    
    # Prepare validation data
    X_val = None
    y_val = None
    if df_val is not None:
        y_val = model._create_surge_target(df_val)
        X_val = df_val[feature_columns]
    
    # Fit model
    model.fit(X_train, y_train, X_val, y_val)
    
    # Cross-validation
    cv_results = model.cross_validate(X_train, y_train)
    
    # Generate training report
    training_report = {
        'model_type': model_type,
        'quantile_alpha': quantile_alpha,
        'feature_columns': feature_columns,
        'feature_importance': model.get_feature_importance(),
        'cv_results': cv_results,
        'target_stats': {
            'mean_surge_multiplier': float(y_train.mean()),
            'std_surge_multiplier': float(y_train.std()),
            'min_surge_multiplier': float(y_train.min()),
            'max_surge_multiplier': float(y_train.max())
        }
    }
    
    # Add validation metrics if available
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        training_report['validation_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'pinball': mean_pinball_loss(y_val, val_pred, alpha=quantile_alpha)
        }
    
    logger.info("Surge model training completed")
    
    return model, training_report

def create_ensemble_surge_model(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None
) -> Tuple[Dict[str, SurgeModel], Dict[str, Any]]:
    """
    Create an ensemble of surge models for different quantiles
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame (optional)
        feature_columns: List of feature columns to use
    
    Returns:
        Tuple of (ensemble models, training report)
    """
    logger.info("Creating ensemble surge model")
    
    # Define quantiles for ensemble
    quantiles = [0.1, 0.5, 0.9]  # Lower bound, median, upper bound
    model_types = ['random_forest', 'gradient_boosting']
    
    ensemble_models = {}
    ensemble_report = {
        'quantiles': quantiles,
        'models': {},
        'ensemble_performance': {}
    }
    
    for quantile in quantiles:
        for model_type in model_types:
            model_name = f"{model_type}_q{quantile}"
            
            try:
                model, report = train_surge_model(
                    df_train, df_val, feature_columns, 
                    model_type=model_type, quantile_alpha=quantile
                )
                
                ensemble_models[model_name] = model
                ensemble_report['models'][model_name] = report
                
                logger.info(f"Trained {model_name}: CV RMSE = {report['cv_results']['mean_rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                ensemble_report['models'][model_name] = {'error': str(e)}
    
    # Select best model for each quantile
    best_models = {}
    for quantile in quantiles:
        best_rmse = float('inf')
        best_model_name = None
        
        for model_name, report in ensemble_report['models'].items():
            if f'_q{quantile}' in model_name and 'cv_results' in report:
                rmse = report['cv_results']['mean_rmse']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = model_name
        
        if best_model_name:
            best_models[f'q{quantile}'] = ensemble_models[best_model_name]
            ensemble_report['ensemble_performance'][f'q{quantile}'] = {
                'best_model': best_model_name,
                'best_rmse': best_rmse
            }
    
    logger.info("Ensemble surge model training completed")
    
    return best_models, ensemble_report
