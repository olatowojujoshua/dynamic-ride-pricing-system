"""
Regularized baseline model training to address overfitting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
import lightgbm as lgb

from ..config import BASELINE_MODEL_PATH, TARGET_COLUMN, CV_FOLDS, RANDOM_SEED
from ..utils.logger import logger
from ..utils.seed import set_random_seed

class RegularizedBaselineModel:
    """
    Regularized baseline model with overfitting prevention
    """
    
    def __init__(self, model_type: str = 'regularized_rf'):
        """
        Initialize the regularized baseline model
        
        Args:
            model_type: Type of model ('regularized_rf', 'regularized_xgb', 'ridge', 'elastic_net')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_fitted = False
        self.best_params = None
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specific model type with regularization"""
        set_random_seed(RANDOM_SEED)
        
        if self.model_type == 'regularized_rf':
            # Heavily regularized Random Forest
            self.model = RandomForestRegressor(
                n_estimators=50,  # Reduced from 100
                max_depth=6,      # Reduced from 10
                min_samples_split=10,  # Increased from 5
                min_samples_leaf=5,    # Increased from 2
                max_features='sqrt',   # Added feature restriction
                bootstrap=True,
                oob_score=True,       # Enable out-of-bag scoring
                random_state=RANDOM_SEED
            )
        elif self.model_type == 'regularized_xgb':
            # Regularized XGBoost
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,           # Reduced depth
                learning_rate=0.05,    # Lower learning rate
                subsample=0.8,         # Row subsampling
                colsample_bytree=0.8,  # Column subsampling
                reg_alpha=0.1,         # L1 regularization
                reg_lambda=1.0,        # L2 regularization
                random_state=RANDOM_SEED
            )
        elif self.model_type == 'ridge':
            # Ridge regression with strong regularization
            self.model = Ridge(alpha=10.0, random_state=RANDOM_SEED)
        elif self.model_type == 'elastic_net':
            # Elastic Net for feature selection
            self.model = ElasticNet(
                alpha=0.1, 
                l1_ratio=0.5,
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def hyperparameter_tune(self, X_train, y_train):
        """
        Perform hyperparameter tuning with cross-validation
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        
        if self.model_type == 'regularized_rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            }
        elif self.model_type == 'regularized_xgb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'reg_alpha': [0.01, 0.1, 1.0],
                'reg_lambda': [0.1, 1.0, 10.0]
            }
        elif self.model_type == 'ridge':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        elif self.model_type == 'elastic_net':
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        
        # Use TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, -grid_search.best_score_
    
    def train(self, X_train, y_train, tune_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train the regularized model
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparams: Whether to perform hyperparameter tuning
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training regularized baseline model: {self.model_type}")
        
        # Scale features for linear models
        if self.model_type in ['ridge', 'elastic_net']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # Hyperparameter tuning
        if tune_hyperparams:
            self.hyperparameter_tune(X_train, y_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Training predictions
        y_train_pred = self.model.predict(X_train)
        
        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Cross-validation with temporal split
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                   cv=tscv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Feature importance (if available)
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(X_train.columns, np.abs(self.model.coef_)))
        
        # Out-of-bag score (for Random Forest)
        oob_score = None
        if hasattr(self.model, 'oob_score_'):
            oob_score = self.model.oob_score_
        
        training_metrics = {
            'model_type': self.model_type,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_rmse_scores': cv_rmse.tolist(),
            'feature_importance': feature_importance,
            'best_params': self.best_params,
            'oob_score': oob_score,
            'overfitting_indicator': train_r2 - (1 - cv_rmse.mean() / np.std(y_train))  # Overfitting metric
        }
        
        logger.info(f"Training metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        logger.info(f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        
        if oob_score:
            logger.info(f"OOB Score: {oob_score:.4f}")
        
        # Check for overfitting
        if training_metrics['overfitting_indicator'] > 0.3:
            logger.warning("High overfitting detected! Consider increasing regularization.")
        elif training_metrics['overfitting_indicator'] < 0.1:
            logger.info("Low overfitting detected. Model generalizes well.")
        
        return training_metrics
    
    def predict(self, X) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features for linear models
        if self.model_type in ['ridge', 'elastic_net']:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str = BASELINE_MODEL_PATH):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved regularized baseline model to {filepath}")
    
    def load_model(self, filepath: str = BASELINE_MODEL_PATH):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        self.best_params = model_data['best_params']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Loaded regularized baseline model from {filepath}")

def train_regularized_baseline_model(X_train, y_train, model_type: str = 'regularized_rf', 
                                    tune_hyperparams: bool = True) -> Tuple[RegularizedBaselineModel, Dict[str, Any]]:
    """
    Train a regularized baseline model
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train
        tune_hyperparams: Whether to perform hyperparameter tuning
    
    Returns:
        Tuple of (trained model, training metrics)
    """
    model = RegularizedBaselineModel(model_type=model_type)
    metrics = model.train(X_train, y_train, tune_hyperparams=tune_hyperparams)
    
    return model, metrics
