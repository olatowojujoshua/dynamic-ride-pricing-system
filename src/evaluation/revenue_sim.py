"""
Revenue simulation and business impact analysis for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import TARGET_COLUMN
from ..utils.logger import logger

class RevenueSimulator:
    """
    Simulate revenue impact of different pricing strategies
    """
    
    def __init__(self):
        """Initialize revenue simulator"""
        logger.info("Initialized revenue simulator")
    
    def simulate_baseline_vs_dynamic_pricing(self, df: pd.DataFrame,
                                           baseline_prices: np.ndarray,
                                           dynamic_prices: np.ndarray,
                                           demand_elasticity: float = -0.5) -> Dict[str, Any]:
        """
        Simulate revenue comparison between baseline and dynamic pricing
        
        Args:
            df: DataFrame with ride data
            baseline_prices: Array of baseline prices
            dynamic_prices: Array of dynamic prices
            demand_elasticity: Price elasticity of demand
        
        Returns:
            Dictionary with simulation results
        """
        if len(baseline_prices) != len(df) or len(dynamic_prices) != len(df):
            raise ValueError("Price arrays must match DataFrame length")
        
        # Calculate demand changes based on price elasticity
        price_change_ratio = dynamic_prices / baseline_prices
        demand_change_ratio = price_change_ratio ** demand_elasticity
        
        # Simulate demand (number of rides)
        original_demand = np.ones(len(df))  # Assume 1 ride per row in original data
        simulated_demand = original_demand * demand_change_ratio
        
        # Calculate revenues
        baseline_revenue = (baseline_prices * original_demand).sum()
        dynamic_revenue = (dynamic_prices * simulated_demand).sum()
        
        # Calculate metrics
        revenue_lift = (dynamic_revenue / baseline_revenue - 1) * 100
        demand_change = (simulated_demand.sum() / original_demand.sum() - 1) * 100
        price_change = (dynamic_prices.mean() / baseline_prices.mean() - 1) * 100
        
        results = {
            'baseline_revenue': float(baseline_revenue),
            'dynamic_revenue': float(dynamic_revenue),
            'revenue_lift_percentage': float(revenue_lift),
            'demand_change_percentage': float(demand_change),
            'price_change_percentage': float(price_change),
            'total_rides_baseline': float(original_demand.sum()),
            'total_rides_dynamic': float(simulated_demand.sum()),
            'average_price_baseline': float(baseline_prices.mean()),
            'average_price_dynamic': float(dynamic_prices.mean()),
            'demand_elasticity_used': demand_elasticity
        }
        
        # Additional analysis
        results['price_distribution'] = {
            'baseline_min': float(baseline_prices.min()),
            'baseline_max': float(baseline_prices.max()),
            'baseline_std': float(baseline_prices.std()),
            'dynamic_min': float(dynamic_prices.min()),
            'dynamic_max': float(dynamic_prices.max()),
            'dynamic_std': float(dynamic_prices.std())
        }
        
        # Revenue by segments
        if 'Location_Category' in df.columns:
            results['revenue_by_location'] = self._calculate_segment_revenue(
                df, baseline_prices, dynamic_prices, simulated_demand, 'Location_Category'
            )
        
        if 'Time_of_Booking' in df.columns:
            results['revenue_by_time'] = self._calculate_segment_revenue(
                df, baseline_prices, dynamic_prices, simulated_demand, 'Time_of_Booking'
            )
        
        logger.info(f"Revenue simulation: {revenue_lift:.1f}% lift, {demand_change:.1f}% demand change")
        
        return results
    
    def _calculate_segment_revenue(self, df: pd.DataFrame, baseline_prices: np.ndarray,
                                 dynamic_prices: np.ndarray, simulated_demand: np.ndarray,
                                 segment_col: str) -> Dict[str, Any]:
        """
        Calculate revenue by segment
        
        Args:
            df: DataFrame with segment data
            baseline_prices: Baseline prices
            dynamic_prices: Dynamic prices
            simulated_demand: Simulated demand
            segment_col: Column name for segmentation
        
        Returns:
            Dictionary with segment-wise revenue
        """
        segment_results = {}
        
        for segment in df[segment_col].unique():
            mask = df[segment_col] == segment
            
            segment_baseline_revenue = (baseline_prices[mask]).sum()
            segment_dynamic_revenue = (dynamic_prices[mask] * simulated_demand[mask]).sum()
            
            segment_lift = (segment_dynamic_revenue / segment_baseline_revenue - 1) * 100 if segment_baseline_revenue > 0 else 0
            
            segment_results[segment] = {
                'baseline_revenue': float(segment_baseline_revenue),
                'dynamic_revenue': float(segment_dynamic_revenue),
                'revenue_lift_percentage': float(segment_lift),
                'ride_count': int(mask.sum())
            }
        
        return segment_results
    
    def simulate_pricing_scenarios(self, df: pd.DataFrame,
                                  scenario_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate multiple pricing scenarios
        
        Args:
            df: DataFrame with ride data
            scenario_configs: Dictionary with scenario configurations
        
        Returns:
            Dictionary with scenario comparison results
        """
        results = {
            'scenarios': {},
            'comparison': {},
            'best_scenario': None
        }
        
        # Get baseline prices (actual historical prices)
        baseline_prices = df[TARGET_COLUMN].values
        
        for scenario_name, config in scenario_configs.items():
            # Generate scenario prices
            scenario_prices = self._generate_scenario_prices(df, config)
            
            # Simulate revenue
            scenario_results = self.simulate_baseline_vs_dynamic_pricing(
                df, baseline_prices, scenario_prices, 
                config.get('demand_elasticity', -0.5)
            )
            
            results['scenarios'][scenario_name] = scenario_results
            results['scenarios'][scenario_name]['config'] = config
        
        # Compare scenarios
        scenario_revenues = {name: results['scenarios'][name]['dynamic_revenue'] 
                            for name in scenario_configs.keys()}
        
        best_scenario = max(scenario_revenues.items(), key=lambda x: x[1])
        results['best_scenario'] = best_scenario[0]
        
        # Create comparison table
        comparison_data = []
        for scenario_name, scenario_results in results['scenarios'].items():
            comparison_data.append({
                'scenario': scenario_name,
                'revenue_lift': scenario_results['revenue_lift_percentage'],
                'demand_change': scenario_results['demand_change_percentage'],
                'price_change': scenario_results['price_change_percentage']
            })
        
        results['comparison'] = pd.DataFrame(comparison_data).to_dict('records')
        
        logger.info(f"Simulated {len(scenario_configs)} scenarios. Best: {best_scenario[0]}")
        
        return results
    
    def _generate_scenario_prices(self, df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
        """
        Generate prices for a specific scenario
        
        Args:
            df: DataFrame with ride data
            config: Scenario configuration
        
        Returns:
            Array of scenario prices
        """
        base_prices = df[TARGET_COLUMN].values.copy()
        
        # Apply scenario modifications
        scenario_type = config.get('type', 'multiplier')
        
        if scenario_type == 'multiplier':
            # Apply uniform multiplier
            multiplier = config.get('multiplier', 1.0)
            scenario_prices = base_prices * multiplier
        
        elif scenario_type == 'surge_aggressive':
            # More aggressive surge pricing
            surge_multiplier = config.get('surge_multiplier', 1.5)
            demand_supply_ratio = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1e-8)
            
            # Apply surge based on demand-supply ratio
            surge_factors = np.where(demand_supply_ratio > 1.5, surge_multiplier, 1.0)
            scenario_prices = base_prices * surge_factors
        
        elif scenario_type == 'time_based':
            # Time-based pricing adjustments
            time_multipliers = {
                'Morning': 1.2,
                'Evening': 1.3,
                'Afternoon': 1.1,
                'Night': 1.0,
                'Early Morning': 0.9,
                'Late Night': 0.8
            }
            
            scenario_prices = base_prices.copy()
            for time_period, multiplier in time_multipliers.items():
                mask = df['Time_of_Booking'] == time_period
                scenario_prices[mask] *= multiplier
        
        elif scenario_type == 'loyalty_focused':
            # Loyalty-focused pricing
            loyalty_discounts = {
                'Silver': 0.95,
                'Gold': 0.9,
                'Platinum': 0.85
            }
            
            scenario_prices = base_prices.copy()
            for loyalty_status, discount in loyalty_discounts.items():
                mask = df['Customer_Loyalty_Status'] == loyalty_status
                scenario_prices[mask] *= discount
        
        else:
            # Default: no change
            scenario_prices = base_prices
        
        return scenario_prices
    
    def calculate_break_even_analysis(self, df: pd.DataFrame,
                                    implementation_cost: float,
                                    price_changes: np.ndarray) -> Dict[str, Any]:
        """
        Calculate break-even analysis for pricing system implementation
        
        Args:
            df: DataFrame with ride data
            implementation_cost: One-time implementation cost
            price_changes: Array of price changes (new - old)
        
        Returns:
            Dictionary with break-even analysis
        """
        # Calculate additional revenue per ride
        additional_revenue_per_ride = price_changes
        
        # Simulate demand response
        demand_elasticity = -0.5  # Default elasticity
        price_change_ratio = (df[TARGET_COLUMN].values + price_changes) / df[TARGET_COLUMN].values
        demand_response = price_change_ratio ** demand_elasticity
        
        # Calculate net additional revenue
        net_additional_revenue = additional_revenue_per_ride * demand_response
        
        # Calculate daily/weekly/monthly revenue
        total_additional_revenue = net_additional_revenue.sum()
        
        # Break-even calculation
        if total_additional_revenue > 0:
            rides_per_day = len(df) / 30  # Assume 30 days of data
            daily_additional_revenue = total_additional_revenue / 30
            break_even_days = implementation_cost / daily_additional_revenue
            break_even_months = break_even_days / 30
        else:
            break_even_days = float('inf')
            break_even_months = float('inf')
        
        results = {
            'implementation_cost': float(implementation_cost),
            'total_additional_revenue': float(total_additional_revenue),
            'average_additional_revenue_per_ride': float(net_additional_revenue.mean()),
            'daily_additional_revenue': float(total_additional_revenue / 30),
            'break_even_days': break_even_days,
            'break_even_months': break_even_months,
            'roi_first_year': float((total_additional_revenue * 12 - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else 0
        }
        
        return results
    
    def simulate_long_term_impact(self, df: pd.DataFrame,
                               monthly_growth_rate: float = 0.02,
                               months: int = 12) -> Dict[str, Any]:
        """
        Simulate long-term impact of dynamic pricing
        
        Args:
            df: DataFrame with current ride data
            monthly_growth_rate: Monthly growth rate in ride volume
            months: Number of months to simulate
        
        Returns:
            Dictionary with long-term simulation results
        """
        current_monthly_rides = len(df)
        current_monthly_revenue = df[TARGET_COLUMN].sum()
        
        # Assume dynamic pricing increases revenue by 15% initially
        dynamic_pricing_lift = 0.15
        
        projections = []
        cumulative_revenue = 0
        
        for month in range(1, months + 1):
            # Project ride growth
            projected_rides = current_monthly_rides * ((1 + monthly_growth_rate) ** month)
            
            # Project revenue with dynamic pricing impact
            base_revenue = current_monthly_revenue * ((1 + monthly_growth_rate) ** month)
            dynamic_revenue = base_revenue * (1 + dynamic_pricing_lift)
            
            # Assume dynamic pricing impact diminishes slightly over time
            dynamic_pricing_lift *= 0.98  # 2% reduction per month
            
            cumulative_revenue += dynamic_revenue
            
            projections.append({
                'month': month,
                'projected_rides': int(projected_rides),
                'base_revenue': float(base_revenue),
                'dynamic_revenue': float(dynamic_revenue),
                'revenue_lift': float(dynamic_revenue - base_revenue),
                'cumulative_revenue': float(cumulative_revenue)
            })
        
        # Calculate summary metrics
        total_base_revenue = sum(p['base_revenue'] for p in projections)
        total_dynamic_revenue = sum(p['dynamic_revenue'] for p in projections)
        total_lift = total_dynamic_revenue - total_base_revenue
        
        results = {
            'projections': projections,
            'summary': {
                'total_base_revenue_12_months': float(total_base_revenue),
                'total_dynamic_revenue_12_months': float(total_dynamic_revenue),
                'total_revenue_lift_12_months': float(total_lift),
                'average_monthly_lift': float(total_lift / months),
                'total_rides_12_months': sum(p['projected_rides'] for p in projections)
            }
        }
        
        return results
    
    def generate_revenue_report(self, simulation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive revenue report
        
        Args:
            simulation_results: Results from revenue simulation
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== DYNAMIC PRICING REVENUE IMPACT REPORT ===\n")
        
        # Key metrics
        if 'revenue_lift_percentage' in simulation_results:
            report.append(f"Revenue Lift: {simulation_results['revenue_lift_percentage']:.1f}%")
            report.append(f"Demand Change: {simulation_results['demand_change_percentage']:.1f}%")
            report.append(f"Price Change: {simulation_results['price_change_percentage']:.1f}%")
            report.append("")
        
        # Revenue comparison
        if 'baseline_revenue' in simulation_results:
            report.append("REVENUE COMPARISON:")
            report.append(f"Baseline Revenue: ${simulation_results['baseline_revenue']:,.2f}")
            report.append(f"Dynamic Revenue: ${simulation_results['dynamic_revenue']:,.2f}")
            report.append("")
        
        # Segment analysis
        if 'revenue_by_location' in simulation_results:
            report.append("REVENUE BY LOCATION:")
            for location, metrics in simulation_results['revenue_by_location'].items():
                report.append(f"  {location}: {metrics['revenue_lift_percentage']:.1f}% lift")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        lift = simulation_results.get('revenue_lift_percentage', 0)
        if lift > 10:
            report.append("✓ Dynamic pricing shows strong revenue potential")
            report.append("✓ Consider gradual implementation to monitor customer response")
        elif lift > 5:
            report.append("✓ Dynamic pricing shows moderate revenue potential")
            report.append("✓ Consider A/B testing before full rollout")
        else:
            report.append("⚠ Revenue lift is minimal - review pricing strategy")
            report.append("⚠ Consider adjusting elasticity assumptions")
        
        return "\n".join(report)
