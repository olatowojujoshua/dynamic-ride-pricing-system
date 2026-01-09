"""
Prediction interface for Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .train_baseline import BaselineModel
from .train_surge import SurgeModel
from ..config import TARGET_COLUMN
from ..utils.logger import logger

class PricingPredictor:
    """
    Unified interface for making pricing predictions
    """
    
    def __init__(self, 
                 baseline_model: Optional[BaselineModel] = None,
                 surge_model: Optional[SurgeModel] = None):
        """
        Initialize pricing predictor
        
        Args:
            baseline_model: Trained baseline model
            surge_model: Trained surge model
        """
        self.baseline_model = baseline_model
        self.surge_model = surge_model
        logger.info("Initialized pricing predictor")
    
    def predict_baseline_price(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict baseline price using baseline model
        
        Args:
            features: DataFrame with features
        
        Returns:
            Array of predicted baseline prices
        """
        if self.baseline_model is None:
            raise ValueError("Baseline model not loaded")
        
        if not self.baseline_model.is_fitted:
            raise ValueError("Baseline model not fitted")
        
        return self.baseline_model.predict(features)
    
    def predict_surge_multiplier(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict surge multiplier using surge model
        
        Args:
            features: DataFrame with features
        
        Returns:
            Array of predicted surge multipliers
        """
        if self.surge_model is None:
            raise ValueError("Surge model not loaded")
        
        if not self.surge_model.is_fitted:
            raise ValueError("Surge model not fitted")
        
        return self.surge_model.predict(features)
    
    def predict_final_price(self, 
                         baseline_features: pd.DataFrame,
                         surge_features: pd.DataFrame,
                         constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Predict final price combining baseline and surge predictions
        
        Args:
            baseline_features: Features for baseline model
            surge_features: Features for surge model
            constraints: Optional pricing constraints
        
        Returns:
            Array of final prices
        """
        # Get baseline predictions
        baseline_prices = self.predict_baseline_price(baseline_features)
        
        # Get surge predictions
        surge_multipliers = self.predict_surge_multiplier(surge_features)
        
        # Calculate final prices
        final_prices = baseline_prices * surge_multipliers
        
        # Apply constraints if provided
        if constraints:
            min_price = constraints.get('min_price', 0)
            max_price = constraints.get('max_price', float('inf'))
            final_prices = np.clip(final_prices, min_price, max_price)
        
        return final_prices
    
    def predict_with_confidence(self,
                              baseline_features: pd.DataFrame,
                              surge_features: pd.DataFrame,
                              confidence_level: float = 0.9) -> Dict[str, np.ndarray]:
        """
        Predict with confidence intervals
        
        Args:
            baseline_features: Features for baseline model
            surge_features: Features for surge model
            confidence_level: Confidence level for intervals
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Get point predictions
        baseline_prices = self.predict_baseline_price(baseline_features)
        surge_multipliers = self.predict_surge_multiplier(surge_features)
        final_prices = baseline_prices * surge_multipliers
        
        # Get confidence intervals for surge model if available
        if self.surge_model and hasattr(self.surge_model, 'predict_interval'):
            median, lower_bound, upper_bound = self.surge_model.predict_interval(surge_features)
            
            # Calculate confidence intervals for final prices
            lower_prices = baseline_prices * lower_bound
            upper_prices = baseline_prices * upper_bound
            
            return {
                'final_price': final_prices,
                'baseline_price': baseline_prices,
                'surge_multiplier': surge_multipliers,
                'lower_bound': lower_prices,
                'upper_bound': upper_prices,
                'confidence_level': confidence_level
            }
        else:
            # Simple confidence interval estimation
            std_estimate = np.std(final_prices) * 0.1  # Rough estimate
            lower_prices = final_prices - 1.96 * std_estimate
            upper_prices = final_prices + 1.96 * std_estimate
            
            return {
                'final_price': final_prices,
                'baseline_price': baseline_prices,
                'surge_multiplier': surge_multipliers,
                'lower_bound': lower_prices,
                'upper_bound': upper_prices,
                'confidence_level': confidence_level
            }
    
    def batch_predict(self, 
                    requests: list,
                    feature_extractor) -> list:
        """
        Batch predict for multiple requests
        
        Args:
            requests: List of pricing requests
            feature_extractor: Function to extract features from requests
        
        Returns:
            List of prediction results
        """
        results = []
        
        for request in requests:
            try:
                # Extract features
                baseline_features, surge_features = feature_extractor(request)
                
                # Make prediction
                prediction = self.predict_final_price(baseline_features, surge_features)
                
                results.append({
                    'request_id': getattr(request, 'request_id', 'unknown'),
                    'predicted_price': float(prediction[0]) if len(prediction) > 0 else 0,
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                logger.error(f"Error processing request {getattr(request, 'request_id', 'unknown')}: {str(e)}")
                results.append({
                    'request_id': getattr(request, 'request_id', 'unknown'),
                    'predicted_price': 0,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'baseline_model_loaded': self.baseline_model is not None,
            'surge_model_loaded': self.surge_model is not None,
            'baseline_model_fitted': self.baseline_model.is_fitted if self.baseline_model else False,
            'surge_model_fitted': self.surge_model.is_fitted if self.surge_model else False
        }
        
        if self.baseline_model and self.baseline_model.is_fitted:
            info['baseline_model_type'] = self.baseline_model.model_type
            info['baseline_feature_importance'] = self.baseline_model.get_feature_importance()
        
        if self.surge_model and self.surge_model.is_fitted:
            info['surge_model_type'] = self.surge_model.model_type
            info['surge_feature_importance'] = self.surge_model.get_feature_importance()
        
        return info
