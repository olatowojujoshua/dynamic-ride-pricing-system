"""
Stability analysis for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import logger

class StabilityAnalyzer:
    """
    Analyze price stability and volatility in the dynamic pricing system
    """
    
    def __init__(self):
        """Initialize stability analyzer"""
        logger.info("Initialized stability analyzer")
    
    def calculate_price_volatility(self, prices: np.ndarray, 
                                  time_periods: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate price volatility metrics
        
        Args:
            prices: Array of prices
            time_periods: Optional array of time periods for grouping
        
        Returns:
            Dictionary with volatility metrics
        """
        if len(prices) == 0:
            return {'error': 'Empty price array'}
        
        volatility_metrics = {
            'price_std': float(np.std(prices)),
            'price_variance': float(np.var(prices)),
            'price_range': float(np.max(prices) - np.min(prices)),
            'coefficient_of_variation': float(np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0,
            'price_skewness': float(stats.skew(prices)),
            'price_kurtosis': float(stats.kurtosis(prices)),
            'interquartile_range': float(np.percentile(prices, 75) - np.percentile(prices, 25)),
            'median_absolute_deviation': float(np.median(np.abs(prices - np.median(prices))))
        }
        
        # Rolling volatility if time periods are provided
        if time_periods is not None and len(time_periods) == len(prices):
            volatility_metrics['rolling_volatility'] = self._calculate_rolling_volatility(prices, time_periods)
        
        return volatility_metrics
    
    def _calculate_rolling_volatility(self, prices: np.ndarray, 
                                    time_periods: np.ndarray, 
                                    window_size: int = 10) -> Dict[str, float]:
        """
        Calculate rolling volatility metrics
        
        Args:
            prices: Array of prices
            time_periods: Array of time periods
            window_size: Size of rolling window
        
        Returns:
            Dictionary with rolling volatility metrics
        """
        if len(prices) < window_size:
            return {'note': f'Insufficient data for rolling window of size {window_size}'}
        
        # Calculate rolling standard deviation
        rolling_std = []
        for i in range(len(prices) - window_size + 1):
            window_prices = prices[i:i + window_size]
            rolling_std.append(np.std(window_prices))
        
        rolling_std = np.array(rolling_std)
        
        return {
            'mean_rolling_std': float(np.mean(rolling_std)),
            'max_rolling_std': float(np.max(rolling_std)),
            'min_rolling_std': float(np.min(rolling_std)),
            'std_of_rolling_std': float(np.std(rolling_std))
        }
    
    def analyze_price_stability_over_time(self, df: pd.DataFrame,
                                        price_col: str = 'predicted_price',
                                        time_col: str = 'Time_of_Booking') -> Dict[str, Any]:
        """
        Analyze price stability across different time periods
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with prices
            time_col: Column with time periods
        
        Returns:
            Dictionary with time-based stability analysis
        """
        if price_col not in df.columns or time_col not in df.columns:
            raise ValueError(f"Required columns not found: {price_col}, {time_col}")
        
        stability_analysis = {
            'overall_volatility': self.calculate_price_volatility(df[price_col].values),
            'time_period_analysis': {},
            'stability_score': None
        }
        
        # Analyze by time periods
        for time_period in df[time_col].unique():
            period_data = df[df[time_col] == time_period]
            period_prices = period_data[price_col].values
            
            if len(period_prices) > 0:
                period_volatility = self.calculate_price_volatility(period_prices)
                stability_analysis['time_period_analysis'][time_period] = {
                    'volatility': period_volatility,
                    'sample_size': len(period_prices),
                    'mean_price': float(np.mean(period_prices)),
                    'price_range': float(np.max(period_prices) - np.min(period_prices))
                }
        
        # Calculate overall stability score
        stability_analysis['stability_score'] = self._calculate_stability_score(stability_analysis)
        
        return stability_analysis
    
    def _calculate_stability_score(self, stability_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall stability score (0-100, higher is more stable)
        
        Args:
            stability_analysis: Stability analysis results
        
        Returns:
            Stability score
        """
        overall_volatility = stability_analysis['overall_volatility']
        
        # Normalize volatility metrics to 0-100 scale
        cv_score = max(0, 100 - overall_volatility['coefficient_of_variation'] * 50)  # Lower CV is better
        range_score = max(0, 100 - overall_volatility['price_range'] / 10)  # Lower range is better
        mad_score = max(0, 100 - overall_volatility['median_absolute_deviation'] * 20)  # Lower MAD is better
        
        # Time period consistency
        time_period_scores = []
        for period_data in stability_analysis['time_period_analysis'].values():
            period_cv = period_data['volatility']['coefficient_of_variation']
            period_score = max(0, 100 - period_cv * 50)
            time_period_scores.append(period_score)
        
        time_consistency_score = np.mean(time_period_scores) if time_period_scores else 100
        
        # Weighted average
        stability_score = (cv_score * 0.3 + range_score * 0.2 + mad_score * 0.2 + time_consistency_score * 0.3)
        
        return float(stability_score)
    
    def detect_price_anomalies(self, df: pd.DataFrame,
                             price_col: str = 'predicted_price',
                             method: str = 'iqr',
                             threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect price anomalies in the dataset
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with prices
            method: Anomaly detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for anomaly detection
        
        Returns:
            Dictionary with anomaly detection results
        """
        if price_col not in df.columns:
            raise ValueError(f"Price column {price_col} not found")
        
        prices = df[price_col].values
        anomaly_results = {
            'method': method,
            'threshold': threshold,
            'anomalies_detected': 0,
            'anomaly_indices': [],
            'anomaly_percentage': 0,
            'anomaly_summary': {}
        }
        
        if method == 'iqr':
            Q1 = np.percentile(prices, 25)
            Q3 = np.percentile(prices, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            anomaly_mask = (prices < lower_bound) | (prices > upper_bound)
            anomaly_indices = np.where(anomaly_mask)[0]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(prices))
            anomaly_mask = z_scores > threshold
            anomaly_indices = np.where(anomaly_mask)[0]
        
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        # Update results
        anomaly_results['anomalies_detected'] = len(anomaly_indices)
        anomaly_results['anomaly_indices'] = anomaly_indices.tolist()
        anomaly_results['anomaly_percentage'] = (len(anomaly_indices) / len(prices)) * 100
        
        if len(anomaly_indices) > 0:
            anomaly_prices = prices[anomaly_indices]
            anomaly_results['anomaly_summary'] = {
                'min_anomaly_price': float(np.min(anomaly_prices)),
                'max_anomaly_price': float(np.max(anomaly_prices)),
                'mean_anomaly_price': float(np.mean(anomaly_prices)),
                'median_anomaly_price': float(np.median(anomaly_prices))
            }
        
        logger.info(f"Detected {len(anomaly_indices)} price anomalies using {method} method")
        
        return anomaly_results
    
    def analyze_surge_stability(self, df: pd.DataFrame,
                             surge_col: str = 'surge_multiplier',
                             time_col: str = 'Time_of_Booking') -> Dict[str, Any]:
        """
        Analyze surge pricing stability
        
        Args:
            df: DataFrame with surge data
            surge_col: Column with surge multipliers
            time_col: Column with time periods
        
        Returns:
            Dictionary with surge stability analysis
        """
        if surge_col not in df.columns:
            return {'error': f'Surge column {surge_col} not found'}
        
        surge_multipliers = df[surge_col].values
        
        surge_stability = {
            'surge_volatility': self.calculate_price_volatility(surge_multipliers),
            'surge_frequency_analysis': self._analyze_surge_frequency(surge_multipliers),
            'time_based_surge_analysis': {}
        }
        
        # Analyze surge by time periods
        if time_col in df.columns:
            for time_period in df[time_col].unique():
                period_data = df[df[time_col] == time_period]
                period_surge = period_data[surge_col].values
                
                if len(period_surge) > 0:
                    surge_stability['time_based_surge_analysis'][time_period] = {
                        'mean_surge': float(np.mean(period_surge)),
                        'surge_frequency': float((period_surge > 1.2).mean() * 100),
                        'extreme_surge_frequency': float((period_surge > 2.0).mean() * 100),
                        'surge_volatility': float(np.std(period_surge))
                    }
        
        # Calculate surge stability score
        surge_stability['surge_stability_score'] = self._calculate_surge_stability_score(surge_stability)
        
        return surge_stability
    
    def _analyze_surge_frequency(self, surge_multipliers: np.ndarray) -> Dict[str, float]:
        """
        Analyze surge frequency patterns
        
        Args:
            surge_multipliers: Array of surge multipliers
        
        Returns:
            Dictionary with surge frequency analysis
        """
        return {
            'no_surge_frequency': float((surge_multipliers <= 1.0).mean() * 100),
            'mild_surge_frequency': float(((surge_multipliers > 1.0) & (surge_multipliers <= 1.5)).mean() * 100),
            'moderate_surge_frequency': float(((surge_multipliers > 1.5) & (surge_multipliers <= 2.0)).mean() * 100),
            'high_surge_frequency': float(((surge_multipliers > 2.0) & (surge_multipliers <= 3.0)).mean() * 100),
            'extreme_surge_frequency': float((surge_multipliers > 3.0).mean() * 100)
        }
    
    def _calculate_surge_stability_score(self, surge_stability: Dict[str, Any]) -> float:
        """
        Calculate surge stability score (0-100, higher is more stable)
        
        Args:
            surge_stability: Surge stability analysis
        
        Returns:
            Surge stability score
        """
        surge_volatility = surge_stability['surge_volatility']
        surge_frequency = surge_stability['surge_frequency_analysis']
        
        # Volatility score (lower is better)
        volatility_score = max(0, 100 - surge_volatility['coefficient_of_variation'] * 100)
        
        # Frequency score (moderate surge frequency is ideal)
        moderate_surge_freq = surge_frequency['moderate_surge_frequency']
        extreme_surge_freq = surge_frequency['extreme_surge_frequency']
        
        # Ideal: moderate surge around 20-30%, extreme surge < 5%
        frequency_score = max(0, 100 - abs(moderate_surge_freq - 25) * 2 - extreme_surge_freq * 5)
        
        # Overall surge stability score
        surge_stability_score = (volatility_score * 0.6 + frequency_score * 0.4)
        
        return float(surge_stability_score)
    
    def analyze_price_consistency(self, df: pd.DataFrame,
                                price_col: str = 'predicted_price',
                                group_cols: List[str] = None) -> Dict[str, Any]:
        """
        Analyze price consistency across different segments
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with prices
            group_cols: List of columns to group by for consistency analysis
        
        Returns:
            Dictionary with consistency analysis
        """
        if group_cols is None:
            group_cols = ['Location_Category', 'Vehicle_Type', 'Time_of_Booking']
        
        consistency_analysis = {
            'group_consistency': {},
            'overall_consistency_score': None
        }
        
        # Filter available columns
        available_group_cols = [col for col in group_cols if col in df.columns]
        
        for col in available_group_cols:
            group_analysis = self._analyze_group_consistency(df, price_col, col)
            consistency_analysis['group_consistency'][col] = group_analysis
        
        # Calculate overall consistency score
        consistency_analysis['overall_consistency_score'] = self._calculate_overall_consistency_score(
            consistency_analysis['group_consistency']
        )
        
        return consistency_analysis
    
    def _analyze_group_consistency(self, df: pd.DataFrame, price_col: str, 
                                 group_col: str) -> Dict[str, Any]:
        """
        Analyze price consistency within a specific group
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with prices
            group_col: Column to group by
        
        Returns:
            Dictionary with group consistency analysis
        """
        group_stats = {}
        group_volatilities = []
        
        for group_value in df[group_col].unique():
            group_data = df[df[group_col] == group_value]
            group_prices = group_data[price_col].values
            
            if len(group_prices) > 0:
                volatility = self.calculate_price_volatility(group_prices)
                group_stats[group_value] = {
                    'mean_price': float(np.mean(group_prices)),
                    'price_std': float(np.std(group_prices)),
                    'coefficient_of_variation': volatility['coefficient_of_variation'],
                    'sample_size': len(group_prices)
                }
                group_volatilities.append(volatility['coefficient_of_variation'])
        
        # Calculate consistency metrics
        mean_prices = [stats['mean_price'] for stats in group_stats.values()]
        price_range = np.max(mean_prices) - np.min(mean_prices) if mean_prices else 0
        avg_cv = np.mean(group_volatilities) if group_volatilities else 0
        
        return {
            'group_statistics': group_stats,
            'price_range_across_groups': float(price_range),
            'average_cv_within_groups': float(avg_cv),
            'consistency_score': max(0, 100 - price_range - avg_cv * 50)
        }
    
    def _calculate_overall_consistency_score(self, group_consistency: Dict[str, Any]) -> float:
        """
        Calculate overall consistency score across all groups
        
        Args:
            group_consistency: Group consistency analysis
        
        Returns:
            Overall consistency score
        """
        consistency_scores = []
        
        for group_analysis in group_consistency.values():
            if 'consistency_score' in group_analysis:
                consistency_scores.append(group_analysis['consistency_score'])
        
        return float(np.mean(consistency_scores)) if consistency_scores else 100
    
    def generate_stability_report(self, stability_analysis: Dict[str, Any]) -> str:
        """
        Generate a comprehensive stability report
        
        Args:
            stability_analysis: Results from stability analysis
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== PRICE STABILITY ANALYSIS REPORT ===\n")
        
        # Overall stability score
        if 'stability_score' in stability_analysis:
            score = stability_analysis['stability_score']
            report.append(f"Overall Stability Score: {score:.1f}/100")
            
            if score >= 80:
                report.append("✓ Excellent price stability")
            elif score >= 60:
                report.append("✓ Good price stability")
            elif score >= 40:
                report.append("⚠ Moderate price stability - room for improvement")
            else:
                report.append("✗ Poor price stability - requires attention")
            report.append("")
        
        # Volatility metrics
        if 'overall_volatility' in stability_analysis:
            volatility = stability_analysis['overall_volatility']
            report.append("VOLATILITY METRICS:")
            report.append(f"  Coefficient of Variation: {volatility['coefficient_of_variation']:.3f}")
            report.append(f"  Price Range: ${volatility['price_range']:.2f}")
            report.append(f"  Standard Deviation: ${volatility['price_std']:.2f}")
            report.append("")
        
        # Time period analysis
        if 'time_period_analysis' in stability_analysis:
            report.append("TIME PERIOD STABILITY:")
            for period, analysis in stability_analysis['time_period_analysis'].items():
                cv = analysis['volatility']['coefficient_of_variation']
                report.append(f"  {period}: CV = {cv:.3f} (Sample: {analysis['sample_size']})")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        score = stability_analysis.get('stability_score', 50)
        
        if score < 60:
            report.append("⚠ Consider implementing price smoothing mechanisms")
            report.append("⚠ Review surge pricing thresholds")
            report.append("⚠ Monitor price volatility during peak hours")
        elif score < 80:
            report.append("✓ Continue monitoring price stability")
            report.append("✓ Consider fine-tuning pricing algorithms")
        else:
            report.append("✓ Price stability is well within acceptable ranges")
            report.append("✓ Maintain current pricing strategy")
        
        return "\n".join(report)
