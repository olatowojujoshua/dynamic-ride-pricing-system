"""
Baseline fare model training for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import xgboost as xgb
import lightgbm as lgb

from ..config import BASELINE_MODEL_PATH, TARGET_COLUMN, CV_FOLDS, RANDOM_SEED
from ..utils.logger import logger
from ..utils.seed import set_random_seed

class BaselineModel:
    """
    Baseline fare prediction model using ensemble methods
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the baseline model
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'linear')
        """
        self.model_type = model_type
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
                random_state=RANDOM_SEED
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'linear':
            self.model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Fit the baseline model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        logger.info(f"Training {self.model_type} baseline model")
        
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
        train_r2 = r2_score(y_train, train_pred)
        
        logger.info(f"Training metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            logger.info(f"Validation metrics - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = CV_FOLDS) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            cv: Number of CV folds
        
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Negative MSE because sklearn's cross_val_score uses negative values for loss functions
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)
        
        cv_results = {
            'mean_rmse': cv_rmse.mean(),
            'std_rmse': cv_rmse.std(),
            'rmse_scores': cv_rmse.tolist()
        }
        
        logger.info(f"CV RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
        
        return cv_results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Dictionary with best parameters and score
        """
        logger.info("Performing hyperparameter tuning")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'linear': {
                'alpha': [0.1, 1.0, 10.0]
            }
        }
        
        if self.model_type not in param_grids:
            logger.warning(f"No parameter grid defined for {self.model_type}")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model,
            param_grids[self.model_type],
            cv=CV_FOLDS,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        # Update feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = dict(zip(X.columns, np.abs(self.model.coef_)))
        
        best_score = np.sqrt(-grid_search.best_score_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV RMSE: {best_score:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }
    
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
    
    def save_model(self, filepath: str = BASELINE_MODEL_PATH):
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
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved baseline model to {filepath}")
    
    def load_model(self, filepath: str = BASELINE_MODEL_PATH):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Loaded baseline model from {filepath}")

def train_baseline_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: str = 'random_forest',
    tune_hyperparameters: bool = False
) -> Tuple[BaselineModel, Dict[str, Any]]:
    """
    Train a baseline fare model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_type: Type of model to train
        tune_hyperparameters: Whether to perform hyperparameter tuning
    
    Returns:
        Tuple of (trained model, training report)
    """
    logger.info(f"Training baseline model: {model_type}")
    
    # Initialize model
    model = BaselineModel(model_type=model_type)
    
    # Hyperparameter tuning if requested
    if tune_hyperparameters:
        tuning_results = model.hyperparameter_tuning(X_train, y_train)
    else:
        tuning_results = {}
    
    # Fit model
    model.fit(X_train, y_train, X_val, y_val)
    
    # Cross-validation
    cv_results = model.cross_validate(X_train, y_train)
    
    # Generate training report
    training_report = {
        'model_type': model_type,
        'feature_importance': model.get_feature_importance(),
        'cv_results': cv_results,
        'tuning_results': tuning_results,
        'is_tuned': tune_hyperparameters
    }
    
    # Add validation metrics if available
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        training_report['validation_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'r2': r2_score(y_val, val_pred)
        }
    
    logger.info("Baseline model training completed")
    
    return model, training_report

def evaluate_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_types: list = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple baseline models and compare performance
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_types: List of model types to evaluate
    
    Returns:
        Dictionary with results for each model type
    """
    if model_types is None:
        model_types = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'linear']
    
    logger.info(f"Evaluating {len(model_types)} baseline models")
    
    results = {}
    
    for model_type in model_types:
        try:
            model, report = train_baseline_model(
                X_train, y_train, X_val, y_val, 
                model_type=model_type, 
                tune_hyperparameters=False
            )
            results[model_type] = report
            logger.info(f"Completed {model_type}: RMSE = {report.get('validation_metrics', {}).get('rmse', 'N/A')}")
        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    # Find best model
    best_model = None
    best_rmse = float('inf')
    
    for model_type, result in results.items():
        if 'validation_metrics' in result:
            rmse = result['validation_metrics']['rmse']
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_type
    
    if best_model:
        logger.info(f"Best baseline model: {best_model} with RMSE: {best_rmse:.4f}")
        results['best_model'] = best_model
        results['best_rmse'] = best_rmse
    
    return results
