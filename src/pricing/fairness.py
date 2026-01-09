"""
Fairness metrics and algorithms for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_absolute_error
from ..config import LOYALTY_DISCOUNTS, LOCATION_PREMIUMS
from ..utils.logger import logger

class FairnessMetrics:
    """
    Calculate and monitor fairness metrics across different customer segments
    """
    
    def __init__(self):
        """Initialize fairness metrics calculator"""
        self.fairness_thresholds = {
            'price_disparity_ratio': 1.5,  # Max 50% price difference between groups
            'mae_disparity_threshold': 0.2,  # Max 20% difference in MAE between groups
            'representation_ratio': 0.8  # Minimum representation in training data
        }
        logger.info("Initialized fairness metrics calculator")
    
    def calculate_price_disparity(self, df: pd.DataFrame, 
                                price_col: str = 'predicted_price',
                                group_col: str = 'Location_Category') -> Dict[str, Any]:
        """
        Calculate price disparity across different groups
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with predicted prices
            group_col: Column defining groups for comparison
        
        Returns:
            Dictionary with price disparity metrics
        """
        if price_col not in df.columns or group_col not in df.columns:
            raise ValueError(f"Required columns not found: {price_col}, {group_col}")
        
        group_stats = df.groupby(group_col)[price_col].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Calculate disparity ratios
        mean_prices = group_stats.set_index(group_col)['mean']
        max_mean = mean_prices.max()
        min_mean = mean_prices.min()
        
        disparity_ratio = max_mean / min_mean if min_mean > 0 else float('inf')
        
        # Coefficient of variation across groups
        cv_across_groups = mean_prices.std() / mean_prices.mean() if mean_prices.mean() > 0 else 0
        
        # Statistical significance test
        groups = [group[price_col].values for name, group in df.groupby(group_col)]
        f_stat, p_value = stats.f_oneway(*groups) if len(groups) > 1 else (0, 1)
        
        disparity_metrics = {
            'group_statistics': group_stats.to_dict('records'),
            'disparity_ratio': disparity_ratio,
            'coefficient_of_variation': cv_across_groups,
            'statistical_test': {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05
            },
            'fairness_violation': disparity_ratio > self.fairness_thresholds['price_disparity_ratio']
        }
        
        logger.info(f"Price disparity for {group_col}: ratio={disparity_ratio:.3f}, significant={p_value < 0.05}")
        
        return disparity_metrics
    
    def calculate_loyalty_fairness(self, df: pd.DataFrame,
                                price_col: str = 'predicted_price',
                                loyalty_col: str = 'Customer_Loyalty_Status') -> Dict[str, Any]:
        """
        Calculate fairness metrics across loyalty tiers
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with predicted prices
            loyalty_col: Column with loyalty status
        
        Returns:
            Dictionary with loyalty fairness metrics
        """
        if price_col not in df.columns or loyalty_col not in df.columns:
            raise ValueError(f"Required columns not found: {price_col}, {loyalty_col}")
        
        loyalty_stats = df.groupby(loyalty_col)[price_col].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Calculate expected discounts based on loyalty
        base_avg = loyalty_stats[loyalty_stats[loyalty_col] == 'Silver']['mean'].values[0] if 'Silver' in loyalty_stats[loyalty_col].values else loyalty_stats['mean'].max()
        
        fairness_metrics = {
            'loyalty_statistics': loyalty_stats.to_dict('records'),
            'expected_discounts': LOYALTY_DISCOUNTS,
            'actual_discounts': {},
            'discount_effectiveness': {}
        }
        
        # Calculate actual discounts received
        for tier in ['Silver', 'Gold', 'Platinum']:
            if tier in loyalty_stats[loyalty_col].values:
                tier_avg = loyalty_stats[loyalty_stats[loyalty_col] == tier]['mean'].values[0]
                actual_discount = (base_avg - tier_avg) / base_avg if base_avg > 0 else 0
                fairness_metrics['actual_discounts'][tier] = actual_discount
                
                expected_discount = LOYALTY_DISCOUNTS.get(tier, 0)
                effectiveness = actual_discount / expected_discount if expected_discount > 0 else 0
                fairness_metrics['discount_effectiveness'][tier] = effectiveness
        
        return fairness_metrics
    
    def calculate_location_fairness(self, df: pd.DataFrame,
                                  price_col: str = 'predicted_price',
                                  location_col: str = 'Location_Category') -> Dict[str, Any]:
        """
        Calculate fairness metrics across location categories
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with predicted prices
            location_col: Column with location categories
        
        Returns:
            Dictionary with location fairness metrics
        """
        if price_col not in df.columns or location_col not in df.columns:
            raise ValueError(f"Required columns not found: {price_col}, {location_col}")
        
        location_stats = df.groupby(location_col)[price_col].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Compare with expected location premiums
        base_avg = location_stats[location_stats[location_col] == 'Suburban']['mean'].values[0] if 'Suburban' in location_stats[location_col].values else location_stats['mean'].mean()
        
        fairness_metrics = {
            'location_statistics': location_stats.to_dict('records'),
            'expected_premiums': LOCATION_PREMIUMS,
            'actual_premiums': {},
            'premium_effectiveness': {}
        }
        
        # Calculate actual premiums
        for location in ['Urban', 'Suburban', 'Rural']:
            if location in location_stats[location_col].values:
                location_avg = location_stats[location_stats[location_col] == location]['mean'].values[0]
                actual_premium = location_avg / base_avg if base_avg > 0 else 1.0
                fairness_metrics['actual_premiums'][location] = actual_premium
                
                expected_premium = LOCATION_PREMIUMS.get(location, 1.0)
                effectiveness = actual_premium / expected_premium if expected_premium > 0 else 1.0
                fairness_metrics['premium_effectiveness'][location] = effectiveness
        
        return fairness_metrics
    
    def calculate_temporal_fairness(self, df: pd.DataFrame,
                                  price_col: str = 'predicted_price',
                                  time_col: str = 'Time_of_Booking') -> Dict[str, Any]:
        """
        Calculate fairness metrics across time periods
        
        Args:
            df: DataFrame with pricing data
            price_col: Column with predicted prices
            time_col: Column with time information
        
        Returns:
            Dictionary with temporal fairness metrics
        """
        if price_col not in df.columns or time_col not in df.columns:
            raise ValueError(f"Required columns not found: {price_col}, {time_col}")
        
        time_stats = df.groupby(time_col)[price_col].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Calculate time-based fairness metrics
        fairness_metrics = {
            'time_statistics': time_stats.to_dict('records'),
            'peak_off_peak_ratio': 0,
            'time_variance': 0
        }
        
        # Peak vs off-peak comparison
        peak_times = ['Morning', 'Evening']
        off_peak_times = ['Early Morning', 'Late Night', 'Night']
        
        peak_avg = df[df[time_col].isin(peak_times)][price_col].mean() if any(df[time_col].isin(peak_times)) else 0
        off_peak_avg = df[df[time_col].isin(off_peak_times)][price_col].mean() if any(df[time_col].isin(off_peak_times)) else 0
        
        if off_peak_avg > 0:
            fairness_metrics['peak_off_peak_ratio'] = peak_avg / off_peak_avg
        
        fairness_metrics['time_variance'] = df.groupby(time_col)[price_col].mean().var()
        
        return fairness_metrics
    
    def calculate_overall_fairness_score(self, fairness_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate an overall fairness score from individual metrics
        
        Args:
            fairness_metrics: Dictionary with various fairness metrics
        
        Returns:
            Dictionary with overall fairness scores
        """
        scores = {}
        
        # Price disparity score (lower is better)
        if 'price_disparity' in fairness_metrics:
            disparity_ratio = fairness_metrics['price_disparity'].get('disparity_ratio', 1.0)
            scores['price_disparity_score'] = max(0, 1 - (disparity_ratio - 1) / 2)  # Normalize to 0-1
        
        # Loyalty fairness score
        if 'loyalty_fairness' in fairness_metrics:
            effectiveness = fairness_metrics['loyalty_fairness'].get('discount_effectiveness', {})
            if effectiveness:
                avg_effectiveness = np.mean(list(effectiveness.values()))
                scores['loyalty_fairness_score'] = min(1.0, avg_effectiveness)
        
        # Location fairness score
        if 'location_fairness' in fairness_metrics:
            effectiveness = fairness_metrics['location_fairness'].get('premium_effectiveness', {})
            if effectiveness:
                avg_effectiveness = np.mean(list(effectiveness.values()))
                scores['location_fairness_score'] = min(1.0, 1 / avg_effectiveness if avg_effectiveness > 0 else 1.0)
        
        # Overall fairness score (weighted average)
        if scores:
            weights = {
                'price_disparity_score': 0.4,
                'loyalty_fairness_score': 0.3,
                'location_fairness_score': 0.3
            }
            
            overall_score = sum(scores.get(metric, 0) * weight for metric, weight in weights.items())
            scores['overall_fairness_score'] = overall_score
        
        return scores
    
    def detect_fairness_violations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect potential fairness violations in the dataset
        
        Args:
            df: DataFrame with pricing data
        
        Returns:
            Dictionary with fairness violations by category
        """
        violations = {
            'price_disparity': [],
            'loyalty_fairness': [],
            'location_fairness': [],
            'temporal_fairness': []
        }
        
        # Check price disparity
        if 'Location_Category' in df.columns and 'predicted_price' in df.columns:
            disparity_metrics = self.calculate_price_disparity(df)
            if disparity_metrics['fairness_violation']:
                violations['price_disparity'].append(
                    f"Price disparity ratio {disparity_metrics['disparity_ratio']:.3f} exceeds threshold {self.fairness_thresholds['price_disparity_ratio']:.3f}"
                )
        
        # Check loyalty fairness
        if 'Customer_Loyalty_Status' in df.columns and 'predicted_price' in df.columns:
            loyalty_metrics = self.calculate_loyalty_fairness(df)
            effectiveness = loyalty_metrics.get('discount_effectiveness', {})
            for tier, eff in effectiveness.items():
                if eff < 0.5:  # Less than 50% effectiveness
                    violations['loyalty_fairness'].append(
                        f"Loyalty discount for {tier} tier is only {eff:.1%} effective"
                    )
        
        # Check location fairness
        if 'Location_Category' in df.columns and 'predicted_price' in df.columns:
            location_metrics = self.calculate_location_fairness(df)
            effectiveness = location_metrics.get('premium_effectiveness', {})
            for location, eff in effectiveness.items():
                if eff > 2.0 or eff < 0.5:  # More than 2x or less than 0.5x expected
                    violations['location_fairness'].append(
                        f"Location premium for {location} is {eff:.1f}x expected"
                    )
        
        return violations

class FairnessEngine:
    """
    Engine for applying fairness adjustments to pricing decisions
    """
    
    def __init__(self, fairness_metrics: Optional[FairnessMetrics] = None):
        """
        Initialize fairness engine
        
        Args:
            fairness_metrics: FairnessMetrics instance
        """
        self.fairness_metrics = fairness_metrics or FairnessMetrics()
        self.fairness_adjustments = {
            'loyalty_boost': True,
            'location_balance': True,
            'temporal_smoothing': True
        }
        logger.info("Initialized fairness engine")
    
    def apply_fairness_adjustments(self, pricing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fairness adjustments to a pricing decision
        
        Args:
            pricing_data: Dictionary with pricing information
        
        Returns:
            Updated pricing data with fairness adjustments
        """
        adjusted_data = pricing_data.copy()
        adjustments_applied = []
        
        # Apply loyalty fairness adjustments
        if self.fairness_adjustments['loyalty_boost']:
            adjusted_data, loyalty_adjustment = self._apply_loyalty_fairness(adjusted_data)
            if loyalty_adjustment:
                adjustments_applied.append(loyalty_adjustment)
        
        # Apply location fairness adjustments
        if self.fairness_adjustments['location_balance']:
            adjusted_data, location_adjustment = self._apply_location_fairness(adjusted_data)
            if location_adjustment:
                adjustments_applied.append(location_adjustment)
        
        # Apply temporal fairness adjustments
        if self.fairness_adjustments['temporal_smoothing']:
            adjusted_data, temporal_adjustment = self._apply_temporal_fairness(adjusted_data)
            if temporal_adjustment:
                adjustments_applied.append(temporal_adjustment)
        
        adjusted_data['fairness_adjustments'] = adjustments_applied
        
        return adjusted_data
    
    def _apply_loyalty_fairness(self, pricing_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Apply loyalty-based fairness adjustments
        
        Args:
            pricing_data: Pricing data dictionary
        
        Returns:
            Tuple of (updated data, adjustment description)
        """
        loyalty_status = pricing_data.get('loyalty_status')
        if not loyalty_status or loyalty_status not in LOYALTY_DISCOUNTS:
            return pricing_data, None
        
        current_price = pricing_data.get('final_price', 0)
        expected_discount = LOYALTY_DISCOUNTS[loyalty_status]
        
        # Ensure loyalty customers get their expected discount
        base_price = pricing_data.get('base_price', current_price)
        expected_price = base_price * (1 - expected_discount)
        
        if current_price > expected_price:
            pricing_data['final_price'] = expected_price
            adjustment = f"Applied loyalty fairness: {loyalty_status} discount of {expected_discount:.1%}"
            return pricing_data, adjustment
        
        return pricing_data, None
    
    def _apply_location_fairness(self, pricing_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Apply location-based fairness adjustments
        
        Args:
            pricing_data: Pricing data dictionary
        
        Returns:
            Tuple of (updated data, adjustment description)
        """
        location = pricing_data.get('location_category')
        if not location or location not in LOCATION_PREMIUMS:
            return pricing_data, None
        
        current_price = pricing_data.get('final_price', 0)
        expected_premium = LOCATION_PREMIUMS[location]
        
        # Check if location premium is being applied fairly
        base_price = pricing_data.get('base_price', current_price)
        expected_price = base_price * expected_premium
        
        # Allow some flexibility (±10%)
        tolerance = 0.1
        if abs(current_price - expected_price) / expected_price > tolerance:
            if location == 'Rural' and current_price > expected_price:
                # Reduce prices in rural areas
                pricing_data['final_price'] = expected_price
                adjustment = f"Applied location fairness: Reduced {location} pricing to expected level"
                return pricing_data, adjustment
            elif location == 'Urban' and current_price < expected_price * 0.8:
                # Ensure urban areas get appropriate premium
                pricing_data['final_price'] = expected_price
                adjustment = f"Applied location fairness: Adjusted {location} pricing to expected level"
                return pricing_data, adjustment
        
        return pricing_data, None
    
    def _apply_temporal_fairness(self, pricing_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Apply temporal fairness adjustments
        
        Args:
            pricing_data: Pricing data dictionary
        
        Returns:
            Tuple of (updated data, adjustment description)
        """
        time_features = pricing_data.get('time_features', {})
        is_rush_hour = time_features.get('is_rush_hour', 0)
        
        current_price = pricing_data.get('final_price', 0)
        base_price = pricing_data.get('base_price', current_price)
        
        # Ensure rush hour pricing is reasonable
        if is_rush_hour:
            rush_multiplier = current_price / base_price if base_price > 0 else 1.0
            
            # Cap rush hour multiplier to prevent excessive pricing
            if rush_multiplier > 2.0:
                pricing_data['final_price'] = base_price * 2.0
                adjustment = f"Applied temporal fairness: Capped rush hour multiplier at 2.0x"
                return pricing_data, adjustment
        
        return pricing_data, None
    
    def generate_fairness_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive fairness report
        
        Args:
            df: DataFrame with pricing data
        
        Returns:
            Dictionary with fairness report
        """
        report = {
            'fairness_metrics': {},
            'fairness_scores': {},
            'violations': {},
            'recommendations': []
        }
        
        # Calculate fairness metrics
        if 'Location_Category' in df.columns and 'predicted_price' in df.columns:
            report['fairness_metrics']['price_disparity'] = self.fairness_metrics.calculate_price_disparity(df)
        
        if 'Customer_Loyalty_Status' in df.columns and 'predicted_price' in df.columns:
            report['fairness_metrics']['loyalty_fairness'] = self.fairness_metrics.calculate_loyalty_fairness(df)
        
        if 'Location_Category' in df.columns and 'predicted_price' in df.columns:
            report['fairness_metrics']['location_fairness'] = self.fairness_metrics.calculate_location_fairness(df)
        
        if 'Time_of_Booking' in df.columns and 'predicted_price' in df.columns:
            report['fairness_metrics']['temporal_fairness'] = self.fairness_metrics.calculate_temporal_fairness(df)
        
        # Calculate overall fairness scores
        report['fairness_scores'] = self.fairness_metrics.calculate_overall_fairness_score(report['fairness_metrics'])
        
        # Detect violations
        report['violations'] = self.fairness_metrics.detect_fairness_violations(df)
        
        # Generate recommendations
        report['recommendations'] = self._generate_fairness_recommendations(report)
        
        return report
    
    def _generate_fairness_recommendations(self, fairness_report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on fairness analysis
        
        Args:
            fairness_report: Fairness analysis report
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check overall fairness score
        overall_score = fairness_report['fairness_scores'].get('overall_fairness_score', 1.0)
        if overall_score < 0.7:
            recommendations.append("Overall fairness score is below 0.7. Review pricing policies.")
        
        # Check price disparity
        if fairness_report['violations'].get('price_disparity'):
            recommendations.append("Address price disparity violations across customer segments.")
        
        # Check loyalty fairness
        if fairness_report['violations'].get('loyalty_fairness'):
            recommendations.append("Review loyalty discount effectiveness and implementation.")
        
        # Check location fairness
        if fairness_report['violations'].get('location_fairness'):
            recommendations.append("Balance location-based pricing to ensure fairness.")
        
        # Check temporal fairness
        if fairness_report['violations'].get('temporal_fairness'):
            recommendations.append("Review time-based pricing for fairness concerns.")
        
        return recommendations
    
    def update_fairness_adjustments(self, **kwargs) -> None:
        """
        Update fairness adjustment settings
        
        Args:
            **kwargs: Fairness adjustment settings to update
        """
        for key, value in kwargs.items():
            if key in self.fairness_adjustments:
                self.fairness_adjustments[key] = value
                logger.info(f"Updated fairness adjustment {key}: {value}")
            else:
                logger.warning(f"Unknown fairness adjustment: {key}")
