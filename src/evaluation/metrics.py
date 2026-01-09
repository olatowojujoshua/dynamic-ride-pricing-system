"""
Comprehensive evaluation metrics for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, mean_pinball_loss
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..config import TARGET_COLUMN, TARGET_MAPE, MIN_REVENUE_LIFT
from ..utils.logger import logger

class ModelMetrics:
    """
    Comprehensive model evaluation metrics
    """
    
    def __init__(self):
        """Initialize model metrics calculator"""
        logger.info("Initialized model metrics calculator")
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary with regression metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_error': np.mean(y_true - y_pred),
            'std_error': np.std(y_true - y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'median_absolute_error': np.median(np.abs(y_true - y_pred))
        }
        
        # Additional metrics
        metrics['rmsle'] = np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))
        metrics['mean_absolute_percentage_error'] = metrics['mape'] * 100  # Convert to percentage
        
        # Custom metrics
        metrics['accuracy_within_10_percent'] = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.1) * 100
        metrics['accuracy_within_20_percent'] = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.2) * 100
        
        return metrics
    
    def calculate_quantile_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  alpha: float = 0.9) -> Dict[str, float]:
        """
        Calculate quantile regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            alpha: Quantile alpha
        
        Returns:
            Dictionary with quantile metrics
        """
        metrics = {
            'pinball_loss': mean_pinball_loss(y_true, y_pred, alpha=alpha),
            'quantile_coverage': np.mean(y_true <= y_pred) * 100,
            'quantile_bias': np.mean(y_pred - y_true)
        }
        
        # Check if coverage is close to target
        target_coverage = alpha * 100
        metrics['coverage_error'] = abs(metrics['quantile_coverage'] - target_coverage)
        
        return metrics
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       threshold: float = 1.5) -> Dict[str, float]:
        """
        Calculate classification metrics for surge prediction
        
        Args:
            y_true: True surge multipliers
            y_pred: Predicted surge multipliers
            threshold: Surge threshold for classification
        
        Returns:
            Dictionary with classification metrics
        """
        # Convert to binary classification (surge vs no surge)
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_type: str = 'regression') -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: Type of model ('regression', 'quantile', 'classification')
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        evaluation_results = {
            'model_type': model_type,
            'sample_size': len(y_true),
            'metrics': {},
            'performance_grade': None,
            'recommendations': []
        }
        
        if model_type == 'regression':
            evaluation_results['metrics'] = self.calculate_regression_metrics(y_true, y_pred)
            
            # Grade performance
            mape = evaluation_results['metrics']['mape']
            r2 = evaluation_results['metrics']['r2']
            
            if mape <= TARGET_MAPE and r2 >= 0.8:
                evaluation_results['performance_grade'] = 'A'
            elif mape <= TARGET_MAPE * 1.5 and r2 >= 0.7:
                evaluation_results['performance_grade'] = 'B'
            elif mape <= TARGET_MAPE * 2 and r2 >= 0.6:
                evaluation_results['performance_grade'] = 'C'
            else:
                evaluation_results['performance_grade'] = 'D'
            
            # Generate recommendations
            if mape > TARGET_MAPE:
                evaluation_results['recommendations'].append(f"MAPE ({mape:.3f}) exceeds target ({TARGET_MAPE:.3f})")
            if r2 < 0.7:
                evaluation_results['recommendations'].append(f"R² ({r2:.3f}) is below acceptable threshold (0.7)")
        
        elif model_type == 'quantile':
            evaluation_results['metrics'] = self.calculate_quantile_metrics(y_true, y_pred)
            
            # Grade performance
            coverage_error = evaluation_results['metrics']['coverage_error']
            pinball_loss = evaluation_results['metrics']['pinball_loss']
            
            if coverage_error <= 5 and pinball_loss <= 0.1:
                evaluation_results['performance_grade'] = 'A'
            elif coverage_error <= 10 and pinball_loss <= 0.15:
                evaluation_results['performance_grade'] = 'B'
            elif coverage_error <= 15 and pinball_loss <= 0.2:
                evaluation_results['performance_grade'] = 'C'
            else:
                evaluation_results['performance_grade'] = 'D'
        
        elif model_type == 'classification':
            evaluation_results['metrics'] = self.calculate_classification_metrics(y_true, y_pred)
            
            # Grade performance
            f1 = evaluation_results['metrics']['f1_score']
            
            if f1 >= 0.9:
                evaluation_results['performance_grade'] = 'A'
            elif f1 >= 0.8:
                evaluation_results['performance_grade'] = 'B'
            elif f1 >= 0.7:
                evaluation_results['performance_grade'] = 'C'
            else:
                evaluation_results['performance_grade'] = 'D'
        
        return evaluation_results
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models
        
        Args:
            model_results: Dictionary with results for each model
        
        Returns:
            Dictionary with model comparison results
        """
        comparison = {
            'model_rankings': {},
            'best_model': None,
            'performance_summary': {},
            'statistical_tests': {}
        }
        
        # Extract key metrics for comparison
        model_metrics = {}
        for model_name, results in model_results.items():
            if 'metrics' in results:
                model_metrics[model_name] = results['metrics']
        
        if not model_metrics:
            return comparison
        
        # Rank models by different metrics
        metrics_to_rank = ['rmse', 'mae', 'mape', 'r2']
        
        for metric in metrics_to_rank:
            if all(metric in metrics for metrics in model_metrics.values()):
                values = {name: metrics[metric] for name, metrics in model_metrics.items()}
                
                if metric in ['rmse', 'mae', 'mape']:
                    # Lower is better
                    sorted_models = sorted(values.items(), key=lambda x: x[1])
                else:
                    # Higher is better
                    sorted_models = sorted(values.items(), key=lambda x: x[1], reverse=True)
                
                comparison['model_rankings'][metric] = [model[0] for model in sorted_models]
        
        # Determine best model overall
        model_scores = {}
        for model_name in model_metrics.keys():
            score = 0
            count = 0
            
            for metric in metrics_to_rank:
                if metric in model_metrics[model_name]:
                    rank = comparison['model_rankings'].get(metric, []).index(model_name) + 1
                    score += rank
                    count += 1
            
            if count > 0:
                model_scores[model_name] = score / count
        
        if model_scores:
            best_model = min(model_scores.items(), key=lambda x: x[1])
            comparison['best_model'] = best_model[0]
        
        # Statistical significance tests (if we have predictions)
        # This is a placeholder - in practice you'd need the actual predictions
        comparison['statistical_tests'] = {
            'note': 'Statistical significance tests require raw predictions'
        }
        
        return comparison

class PricingMetrics:
    """
    Business metrics for pricing system evaluation
    """
    
    def __init__(self):
        """Initialize pricing metrics calculator"""
        logger.info("Initialized pricing metrics calculator")
    
    def calculate_revenue_metrics(self, df: pd.DataFrame, 
                                 price_col: str = 'predicted_price',
                                 actual_price_col: str = 'Historical_Cost_of_Ride') -> Dict[str, float]:
        """
        Calculate revenue-related metrics
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with predicted prices
            actual_price_col: Column with actual prices
        
        Returns:
            Dictionary with revenue metrics
        """
        if price_col not in df.columns or actual_price_col not in df.columns:
            raise ValueError(f"Required columns not found: {price_col}, {actual_price_col}")
        
        predicted_revenue = df[price_col].sum()
        actual_revenue = df[actual_price_col].sum()
        
        metrics = {
            'total_predicted_revenue': float(predicted_revenue),
            'total_actual_revenue': float(actual_revenue),
            'revenue_difference': float(predicted_revenue - actual_revenue),
            'revenue_lift_percentage': float((predicted_revenue / actual_revenue - 1) * 100) if actual_revenue > 0 else 0,
            'average_predicted_price': float(df[price_col].mean()),
            'average_actual_price': float(df[actual_price_col].mean()),
            'price_difference_per_ride': float(df[price_col].mean() - df[actual_price_col].mean())
        }
        
        # Revenue efficiency
        metrics['revenue_efficiency'] = float(predicted_revenue / actual_revenue) if actual_revenue > 0 else 1.0
        
        return metrics
    
    def calculate_demand_elasticity(self, df: pd.DataFrame,
                                   price_col: str = 'predicted_price',
                                   demand_col: str = 'Number_of_Riders') -> Dict[str, float]:
        """
        Calculate price-demand elasticity metrics
        
        Args:
            df: DataFrame with pricing and demand data
            price_col: Column with prices
            demand_col: Column with demand
        
        Returns:
            Dictionary with elasticity metrics
        """
        if price_col not in df.columns or demand_col not in df.columns:
            raise ValueError(f"Required columns not found: {price_col}, {demand_col}")
        
        # Calculate elasticity using log-log regression
        valid_data = df[(df[price_col] > 0) & (df[demand_col] > 0)]
        
        if len(valid_data) < 10:
            return {'elasticity': 0, 'r_squared': 0, 'note': 'Insufficient data for elasticity calculation'}
        
        log_price = np.log(valid_data[price_col])
        log_demand = np.log(valid_data[demand_col])
        
        # Simple linear regression for elasticity
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_demand)
        
        metrics = {
            'price_elasticity': float(slope),
            'elasticity_r_squared': float(r_value ** 2),
            'elasticity_p_value': float(p_value),
            'elasticity_std_error': float(std_err),
            'is_elastic': abs(slope) > 1,
            'sample_size': len(valid_data)
        }
        
        return metrics
    
    def calculate_surge_metrics(self, df: pd.DataFrame,
                               surge_col: str = 'surge_multiplier',
                               price_col: str = 'predicted_price') -> Dict[str, float]:
        """
        Calculate surge pricing metrics
        
        Args:
            df: DataFrame with surge data
            surge_col: Column with surge multipliers
            price_col: Column with prices
        
        Returns:
            Dictionary with surge metrics
        """
        if surge_col not in df.columns:
            return {'note': f'Surge column {surge_col} not found'}
        
        metrics = {
            'average_surge_multiplier': float(df[surge_col].mean()),
            'median_surge_multiplier': float(df[surge_col].median()),
            'max_surge_multiplier': float(df[surge_col].max()),
            'min_surge_multiplier': float(df[surge_col].min()),
            'surge_frequency': float((df[surge_col] > 1.2).mean() * 100),  # % of rides with >20% surge
            'extreme_surge_frequency': float((df[surge_col] > 2.0).mean() * 100)  # % of rides with >100% surge
        }
        
        # Surge distribution
        metrics['surge_std'] = float(df[surge_col].std())
        metrics['surge_coefficient_of_variation'] = metrics['surge_std'] / metrics['average_surge_multiplier'] if metrics['average_surge_multiplier'] > 0 else 0
        
        # Revenue from surge
        if price_col in df.columns:
            base_revenue = (df[price_col] / df[surge_col]).sum()
            surge_revenue = df[price_col].sum()
            metrics['surge_revenue_contribution'] = float((surge_revenue - base_revenue) / surge_revenue * 100) if surge_revenue > 0 else 0
        
        return metrics
    
    def calculate_customer_satisfaction_metrics(self, df: pd.DataFrame,
                                               price_col: str = 'predicted_price',
                                               rating_col: str = 'Average_Ratings') -> Dict[str, float]:
        """
        Calculate customer satisfaction metrics
        
        Args:
            df: DataFrame with pricing and rating data
            price_col: Column with prices
            rating_col: Column with ratings
        
        Returns:
            Dictionary with satisfaction metrics
        """
        if rating_col not in df.columns:
            return {'note': f'Rating column {rating_col} not found'}
        
        metrics = {
            'average_rating': float(df[rating_col].mean()),
            'median_rating': float(df[rating_col].median()),
            'rating_std': float(df[rating_col].std()),
            'high_rating_percentage': float((df[rating_col] >= 4.5).mean() * 100),
            'low_rating_percentage': float((df[rating_col] <= 3.0).mean() * 100)
        }
        
        # Price-rating correlation
        if price_col in df.columns:
            correlation = df[price_col].corr(df[rating_col])
            metrics['price_rating_correlation'] = float(correlation) if not pd.isna(correlation) else 0
        
        return metrics
    
    def calculate_driver_satisfaction_metrics(self, df: pd.DataFrame,
                                            price_col: str = 'predicted_price',
                                            duration_col: str = 'Expected_Ride_Duration') -> Dict[str, float]:
        """
        Calculate driver satisfaction metrics
        
        Args:
            df: DataFrame with pricing and duration data
            price_col: Column with prices
            duration_col: Column with ride duration
        
        Returns:
            Dictionary with driver satisfaction metrics
        """
        metrics = {}
        
        if price_col in df.columns and duration_col in df.columns:
            # Earnings per minute
            metrics['earnings_per_minute'] = float((df[price_col] * 0.75) / df[duration_col]).mean()  # Assume 75% goes to driver
            
            # Minimum earnings compliance
            min_earnings = 10.0  # $10 minimum per ride
            driver_earnings = df[price_col] * 0.75
            metrics['minimum_earnings_compliance'] = float((driver_earnings >= min_earnings).mean() * 100)
        
        return metrics
    
    def evaluate_business_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive business performance evaluation
        
        Args:
            df: DataFrame with all relevant data
        
        Returns:
            Dictionary with business performance metrics
        """
        evaluation = {
            'revenue_metrics': {},
            'surge_metrics': {},
            'customer_satisfaction': {},
            'driver_satisfaction': {},
            'overall_score': None,
            'recommendations': []
        }
        
        # Calculate all metrics
        evaluation['revenue_metrics'] = self.calculate_revenue_metrics(df)
        evaluation['surge_metrics'] = self.calculate_surge_metrics(df)
        evaluation['customer_satisfaction'] = self.calculate_customer_satisfaction_metrics(df)
        evaluation['driver_satisfaction'] = self.calculate_driver_satisfaction_metrics(df)
        
        # Calculate overall score
        scores = []
        
        # Revenue score (0-100)
        revenue_lift = evaluation['revenue_metrics'].get('revenue_lift_percentage', 0)
        if revenue_lift >= MIN_REVENUE_LIFT * 100:
            revenue_score = 100
        elif revenue_lift >= 0:
            revenue_score = (revenue_lift / (MIN_REVENUE_LIFT * 100)) * 100
        else:
            revenue_score = 0
        scores.append(revenue_score)
        
        # Customer satisfaction score (0-100)
        avg_rating = evaluation['customer_satisfaction'].get('average_rating', 0)
        satisfaction_score = min(100, (avg_rating / 5.0) * 100)
        scores.append(satisfaction_score)
        
        # Driver satisfaction score (0-100)
        driver_compliance = evaluation['driver_satisfaction'].get('minimum_earnings_compliance', 0)
        driver_score = driver_compliance
        scores.append(driver_score)
        
        # Overall score
        evaluation['overall_score'] = np.mean(scores)
        
        # Generate recommendations
        if revenue_lift < MIN_REVENUE_LIFT * 100:
            evaluation['recommendations'].append(f"Revenue lift ({revenue_lift:.1f}%) below target ({MIN_REVENUE_LIFT * 100:.1f}%)")
        
        if avg_rating < 4.0:
            evaluation['recommendations'].append(f"Average rating ({avg_rating:.2f}) below target (4.0)")
        
        if driver_compliance < 90:
            evaluation['recommendations'].append(f"Driver minimum earnings compliance ({driver_compliance:.1f}%) below target (90%)")
        
        return evaluation
