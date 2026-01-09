"""
Model calibration utilities for Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from ..utils.logger import logger

class ModelCalibrator:
    """
    Calibrate model predictions for better reliability
    """
    
    def __init__(self, method: str = 'isotonic'):
        """
        Initialize model calibrator
        
        Args:
            method: Calibration method ('isotonic', 'platt', 'beta')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        logger.info(f"Initialized model calibrator with method: {method}")
    
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None):
        """
        Fit calibrator on true values and predictions
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            sample_weight: Optional sample weights
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred, y_true, sample_weight=sample_weight)
        
        elif self.method == 'platt':
            # Platt scaling (logistic regression)
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_pred.reshape(-1, 1), y_true, sample_weight=sample_weight)
        
        elif self.method == 'beta':
            # Beta calibration (simplified version)
            self._fit_beta_calibration(y_true, y_pred, sample_weight)
        
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Fitted {self.method} calibrator on {len(y_true)} samples")
    
    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions
        
        Args:
            y_pred: Raw predictions to calibrate
        
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before making predictions")
        
        if self.method == 'isotonic':
            return self.calibrator.predict(y_pred)
        
        elif self.method == 'platt':
            return self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        
        elif self.method == 'beta':
            return self._predict_beta_calibration(y_pred)
        
        return y_pred
    
    def predict_proba(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities (for probabilistic models)
        
        Args:
            y_pred: Raw predictions
        
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before making predictions")
        
        if self.method == 'isotonic':
            # Isotonic regression doesn't directly provide probabilities
            # Return normalized predictions
            calibrated = self.calibrator.predict(y_pred)
            return (calibrated - calibrated.min()) / (calibrated.max() - calibrated.min())
        
        elif self.method == 'platt':
            return self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        
        elif self.method == 'beta':
            calibrated = self._predict_beta_calibration(y_pred)
            return (calibrated - calibrated.min()) / (calibrated.max() - calibrated.min())
        
        return y_pred
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            sample_weight: Optional[np.ndarray] = None):
        """
        Fit beta calibration parameters
        """
        # Simplified beta calibration using method of moments
        # In practice, you might use more sophisticated methods
        
        # Calculate calibration parameters
        mean_pred = np.mean(y_pred)
        mean_true = np.mean(y_true)
        
        # Simple linear calibration as fallback
        self.beta_params = {
            'alpha': mean_true / mean_pred if mean_pred > 0 else 1.0,
            'beta': 1.0  # Simplified
        }
    
    def _predict_beta_calibration(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Predict using beta calibration
        """
        if not hasattr(self, 'beta_params'):
            return y_pred
        
        alpha = self.beta_params['alpha']
        beta = self.beta_params['beta']
        
        return alpha * (y_pred ** beta)
    
    def evaluate_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          n_bins: int = 10) -> Dict[str, Any]:
        """
        Evaluate calibration quality
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_bins: Number of bins for calibration curve
        
        Returns:
            Dictionary with calibration metrics
        """
        # Get calibrated predictions
        if self.is_fitted:
            y_calibrated = self.predict(y_pred)
        else:
            y_calibrated = y_pred
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_calibrated, n_bins=n_bins, strategy='quantile'
        )
        
        # Calculate Brier score (for probabilistic predictions)
        brier_score = brier_score_loss(y_true, y_calibrated)
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Calculate reliability diagram data
        reliability_data = {
            'bin_edges': np.linspace(0, 1, n_bins + 1),
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
        
        return {
            'calibration_error': calibration_error,
            'brier_score': brier_score,
            'reliability_data': reliability_data,
            'n_bins': n_bins
        }
    
    def plot_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     save_path: Optional[str] = None):
        """
        Plot calibration curve
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Optional path to save plot
        """
        evaluation = self.evaluate_calibration(y_true, y_pred)
        reliability_data = evaluation['reliability_data']
        
        plt.figure(figsize=(10, 6))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot calibration curve
        plt.plot(reliability_data['mean_predicted_value'], 
                reliability_data['fraction_of_positives'], 
                'b-', linewidth=2, label='Model Calibration')
        
        plt.xlabel('Mean Predicted Value')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration plot saved to {save_path}")
        
        plt.show()
    
    def save_calibrator(self, filepath: str):
        """
        Save calibrator to disk
        
        Args:
            filepath: Path to save calibrator
        """
        import joblib
        
        calibrator_data = {
            'method': self.method,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted
        }
        
        if hasattr(self, 'beta_params'):
            calibrator_data['beta_params'] = self.beta_params
        
        joblib.dump(calibrator_data, filepath)
        logger.info(f"Calibrator saved to {filepath}")
    
    def load_calibrator(self, filepath: str):
        """
        Load calibrator from disk
        
        Args:
            filepath: Path to load calibrator from
        """
        import joblib
        
        calibrator_data = joblib.load(filepath)
        
        self.method = calibrator_data['method']
        self.calibrator = calibrator_data['calibrator']
        self.is_fitted = calibrator_data['is_fitted']
        
        if 'beta_params' in calibrator_data:
            self.beta_params = calibrator_data['beta_params']
        
        logger.info(f"Calibrator loaded from {filepath}")

def calibrate_price_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                             method: str = 'isotonic',
                             validation_split: float = 0.2) -> Dict[str, Any]:
    """
    Calibrate price predictions with validation
    
    Args:
        y_true: True price values
        y_pred: Predicted price values
        method: Calibration method
        validation_split: Proportion of data for validation
    
    Returns:
        Dictionary with calibration results
    """
    # Split data for calibration and validation
    split_idx = int(len(y_true) * (1 - validation_split))
    
    y_cal_train, y_cal_val = y_true[:split_idx], y_true[split_idx:]
    y_pred_train, y_pred_val = y_pred[:split_idx], y_pred[split_idx:]
    
    # Fit calibrator
    calibrator = ModelCalibrator(method=method)
    calibrator.fit(y_cal_train, y_pred_train)
    
    # Calibrate validation predictions
    y_pred_cal_val = calibrator.predict(y_pred_val)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    metrics = {
        'method': method,
        'calibration_metrics': calibrator.evaluate_calibration(y_cal_val, y_pred_val),
        'raw_metrics': {
            'mae': mean_absolute_error(y_cal_val, y_pred_val),
            'rmse': np.sqrt(mean_squared_error(y_cal_val, y_pred_val)),
            'mape': np.mean(np.abs((y_cal_val - y_pred_val) / y_cal_val)) * 100
        },
        'calibrated_metrics': {
            'mae': mean_absolute_error(y_cal_val, y_pred_cal_val),
            'rmse': np.sqrt(mean_squared_error(y_cal_val, y_pred_cal_val)),
            'mape': np.mean(np.abs((y_cal_val - y_pred_cal_val) / y_cal_val)) * 100
        }
    }
    
    # Calculate improvement
    metrics['improvement'] = {
        'mae_improvement': (metrics['raw_metrics']['mae'] - metrics['calibrated_metrics']['mae']) / metrics['raw_metrics']['mae'] * 100,
        'rmse_improvement': (metrics['raw_metrics']['rmse'] - metrics['calibrated_metrics']['rmse']) / metrics['raw_metrics']['rmse'] * 100,
        'mape_improvement': (metrics['raw_metrics']['mape'] - metrics['calibrated_metrics']['mape']) / metrics['raw_metrics']['mape'] * 100
    }
    
    return metrics
