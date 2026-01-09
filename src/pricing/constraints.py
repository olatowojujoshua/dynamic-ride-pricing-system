"""
Pricing constraints and business rules for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from ..config import (
    MIN_SURGE_MULTIPLIER, MAX_SURGE_MULTIPLIER, BASE_FLOOR_PRICE, BASE_CEILING_PRICE,
    LOYALTY_DISCOUNTS, LOCATION_PREMIUMS
)
from ..utils.logger import logger

@dataclass
class PricingConstraints:
    """
    Data class for pricing constraints and business rules
    """
    min_surge_multiplier: float = MIN_SURGE_MULTIPLIER
    max_surge_multiplier: float = MAX_SURGE_MULTIPLIER
    base_floor_price: float = BASE_FLOOR_PRICE
    base_ceiling_price: float = BASE_CEILING_PRICE
    loyalty_discounts: Dict[str, float] = None
    location_premiums: Dict[str, float] = None
    max_price_change_rate: float = 0.5  # Max 50% change per hour
    emergency_surge_cap: float = 3.0
    minimum_driver_earnings: float = 10.0  # Minimum earnings per ride
    
    def __post_init__(self):
        if self.loyalty_discounts is None:
            self.loyalty_discounts = LOYALTY_DISCOUNTS.copy()
        if self.location_premiums is None:
            self.location_premiums = LOCATION_PREMIUMS.copy()

class ConstraintManager:
    """
    Manages and applies pricing constraints and business rules
    """
    
    def __init__(self, constraints: Optional[PricingConstraints] = None):
        """
        Initialize the constraint manager
        
        Args:
            constraints: Pricing constraints object
        """
        self.constraints = constraints or PricingConstraints()
        logger.info("Initialized constraint manager")
    
    def apply_surge_constraints(self, surge_multiplier: float) -> float:
        """
        Apply surge multiplier constraints
        
        Args:
            surge_multiplier: Raw surge multiplier
        
        Returns:
            Constrained surge multiplier
        """
        constrained_multiplier = np.clip(
            surge_multiplier,
            self.constraints.min_surge_multiplier,
            self.constraints.max_surge_multiplier
        )
        
        if constrained_multiplier != surge_multiplier:
            logger.debug(f"Constrained surge multiplier: {surge_multiplier:.3f} -> {constrained_multiplier:.3f}")
        
        return constrained_multiplier
    
    def apply_price_bounds(self, price: float) -> float:
        """
        Apply price floor and ceiling constraints
        
        Args:
            price: Raw price
        
        Returns:
            Constrained price
        """
        constrained_price = np.clip(price, self.constraints.base_floor_price, self.constraints.base_ceiling_price)
        
        if constrained_price != price:
            logger.debug(f"Constrained price: ${price:.2f} -> ${constrained_price:.2f}")
        
        return constrained_price
    
    def apply_loyalty_discount(self, price: float, loyalty_status: str) -> float:
        """
        Apply loyalty-based discounts
        
        Args:
            price: Base price
            loyalty_status: Customer loyalty status
        
        Returns:
            Discounted price
        """
        if loyalty_status in self.constraints.loyalty_discounts:
            discount_rate = self.constraints.loyalty_discounts[loyalty_status]
            discounted_price = price * (1 - discount_rate)
            
            logger.debug(f"Applied {loyalty_status} discount: {discount_rate*100:.1f}% -> ${discounted_price:.2f}")
            return discounted_price
        
        return price
    
    def apply_location_premium(self, price: float, location_category: str) -> float:
        """
        Apply location-based price adjustments
        
        Args:
            price: Base price
            location_category: Location category
        
        Returns:
            Adjusted price
        """
        if location_category in self.constraints.location_premiums:
            premium_multiplier = self.constraints.location_premiums[location_category]
            adjusted_price = price * premium_multiplier
            
            logger.debug(f"Applied {location_category} premium: {premium_multiplier:.2f}x -> ${adjusted_price:.2f}")
            return adjusted_price
        
        return price
    
    def apply_time_based_constraints(self, surge_multiplier: float, time_features: Dict[str, Any]) -> float:
        """
        Apply time-based pricing constraints
        
        Args:
            surge_multiplier: Base surge multiplier
            time_features: Dictionary with time-based features
        
        Returns:
            Time-constrained surge multiplier
        """
        constrained_multiplier = surge_multiplier
        
        # Limit surge during off-peak hours
        if time_features.get('is_rush_hour', 0) == 0:
            # Reduce surge during non-rush hours
            constrained_multiplier = min(constrained_multiplier, 1.5)
        
        # Weekend constraints
        if time_features.get('is_weekend', 0) == 1:
            # Allow slightly higher surge on weekends
            constrained_multiplier = min(constrained_multiplier, 2.5)
        
        # Late night constraints
        hour = time_features.get('booking_hour', 12)
        if hour >= 22 or hour <= 5:
            # Limit late night surge for safety reasons
            constrained_multiplier = min(constrained_multiplier, 2.0)
        
        return constrained_multiplier
    
    def apply_driver_earnings_constraint(self, price: float, expected_duration: float) -> float:
        """
        Ensure minimum driver earnings per ride
        
        Args:
            price: Current price
            expected_duration: Expected ride duration in minutes
        
        Returns:
            Price adjusted for minimum earnings
        """
        # Estimate driver earnings (typically 75% of fare)
        driver_earnings = price * 0.75
        
        # Calculate minimum price needed for minimum earnings
        min_price_needed = self.constraints.minimum_driver_earnings / 0.75
        
        if driver_earnings < self.constraints.minimum_driver_earnings:
            adjusted_price = max(price, min_price_needed)
            logger.debug(f"Adjusted for minimum earnings: ${price:.2f} -> ${adjusted_price:.2f}")
            return adjusted_price
        
        return price
    
    def apply_price_stability_constraint(self, current_price: float, previous_price: float, 
                                       time_elapsed_hours: float = 1.0) -> float:
        """
        Apply price stability constraints to prevent rapid price changes
        
        Args:
            current_price: Newly calculated price
            previous_price: Previous price
            time_elapsed_hours: Hours since previous price
        
        Returns:
            Stabilized price
        """
        if previous_price is None or previous_price <= 0:
            return current_price
        
        # Calculate maximum allowed change based on time elapsed
        max_change_rate = self.constraints.max_price_change_rate
        max_allowed_change = previous_price * max_change_rate * time_elapsed_hours
        
        price_change = current_price - previous_price
        
        if abs(price_change) > max_allowed_change:
            # Limit the change
            if price_change > 0:
                stabilized_price = previous_price + max_allowed_change
            else:
                stabilized_price = previous_price - max_allowed_change
            
            logger.debug(f"Stabilized price change: ${current_price:.2f} -> ${stabilized_price:.2f}")
            return stabilized_price
        
        return current_price
    
    def apply_emergency_constraints(self, surge_multiplier: float, emergency_conditions: Dict[str, bool]) -> float:
        """
        Apply emergency pricing constraints
        
        Args:
            surge_multiplier: Base surge multiplier
            emergency_conditions: Dictionary of emergency condition flags
        
        Returns:
            Emergency-constrained surge multiplier
        """
        constrained_multiplier = surge_multiplier
        
        # Check for various emergency conditions
        if emergency_conditions.get('extreme_weather', False):
            # Allow higher surge during extreme weather
            constrained_multiplier = min(constrained_multiplier, self.constraints.emergency_surge_cap)
        
        if emergency_conditions.get('major_event', False):
            # Allow moderate surge for major events
            constrained_multiplier = min(constrained_multiplier, 2.5)
        
        if emergency_conditions.get('system_outage', False):
            # Limit surge during system outages to maintain trust
            constrained_multiplier = min(constrained_multiplier, 1.8)
        
        return constrained_multiplier
    
    def validate_pricing_decision(self, pricing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a complete pricing decision against all constraints
        
        Args:
            pricing_data: Dictionary with all pricing information
        
        Returns:
            Validation results with any adjustments made
        """
        validation_results = {
            'original_price': pricing_data.get('base_price', 0),
            'adjustments': [],
            'final_price': pricing_data.get('base_price', 0),
            'constraints_applied': []
        }
        
        current_price = pricing_data.get('base_price', 0)
        
        # Apply price bounds
        current_price = self.apply_price_bounds(current_price)
        if current_price != validation_results['original_price']:
            validation_results['adjustments'].append(f"Price bounds: ${validation_results['original_price']:.2f} -> ${current_price:.2f}")
            validation_results['constraints_applied'].append('price_bounds')
        
        # Apply location premium
        location = pricing_data.get('location_category')
        if location:
            price_before_location = current_price
            current_price = self.apply_location_premium(current_price, location)
            if current_price != price_before_location:
                validation_results['adjustments'].append(f"Location premium ({location}): ${price_before_location:.2f} -> ${current_price:.2f}")
                validation_results['constraints_applied'].append('location_premium')
        
        # Apply loyalty discount
        loyalty = pricing_data.get('loyalty_status')
        if loyalty:
            price_before_loyalty = current_price
            current_price = self.apply_loyalty_discount(current_price, loyalty)
            if current_price != price_before_loyalty:
                validation_results['adjustments'].append(f"Loyalty discount ({loyalty}): ${price_before_loyalty:.2f} -> ${current_price:.2f}")
                validation_results['constraints_applied'].append('loyalty_discount')
        
        # Apply driver earnings constraint
        duration = pricing_data.get('expected_duration', 0)
        if duration > 0:
            price_before_earnings = current_price
            current_price = self.apply_driver_earnings_constraint(current_price, duration)
            if current_price != price_before_earnings:
                validation_results['adjustments'].append(f"Driver earnings constraint: ${price_before_earnings:.2f} -> ${current_price:.2f}")
                validation_results['constraints_applied'].append('driver_earnings')
        
        validation_results['final_price'] = current_price
        
        return validation_results
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all current constraints
        
        Returns:
            Dictionary with constraint settings
        """
        return {
            'surge_constraints': {
                'min_multiplier': self.constraints.min_surge_multiplier,
                'max_multiplier': self.constraints.max_surge_multiplier,
                'emergency_cap': self.constraints.emergency_surge_cap
            },
            'price_constraints': {
                'floor_price': self.constraints.base_floor_price,
                'ceiling_price': self.constraints.base_ceiling_price,
                'max_change_rate': self.constraints.max_price_change_rate
            },
            'loyalty_discounts': self.constraints.loyalty_discounts,
            'location_premiums': self.constraints.location_premiums,
            'driver_constraints': {
                'minimum_earnings': self.constraints.minimum_driver_earnings
            }
        }
    
    def update_constraints(self, **kwargs) -> None:
        """
        Update constraint values
        
        Args:
            **kwargs: Constraint values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.constraints, key):
                setattr(self.constraints, key, value)
                logger.info(f"Updated constraint {key}: {value}")
            else:
                logger.warning(f"Unknown constraint: {key}")
    
    def check_constraint_violations(self, pricing_data: Dict[str, Any]) -> List[str]:
        """
        Check for any constraint violations in pricing data
        
        Args:
            pricing_data: Dictionary with pricing information
        
        Returns:
            List of constraint violations
        """
        violations = []
        
        price = pricing_data.get('final_price', 0)
        base_price = pricing_data.get('base_price', 0)
        surge_multiplier = pricing_data.get('surge_multiplier', 1.0)
        
        # Check price bounds
        if price < self.constraints.base_floor_price:
            violations.append(f"Price below floor: ${price:.2f} < ${self.constraints.base_floor_price:.2f}")
        
        if price > self.constraints.base_ceiling_price:
            violations.append(f"Price above ceiling: ${price:.2f} > ${self.constraints.base_ceiling_price:.2f}")
        
        # Check surge bounds
        if surge_multiplier < self.constraints.min_surge_multiplier:
            violations.append(f"Surge below minimum: {surge_multiplier:.3f} < {self.constraints.min_surge_multiplier:.3f}")
        
        if surge_multiplier > self.constraints.max_surge_multiplier:
            violations.append(f"Surge above maximum: {surge_multiplier:.3f} > {self.constraints.max_surge_multiplier:.3f}")
        
        # Check driver earnings
        duration = pricing_data.get('expected_duration', 0)
        if duration > 0:
            driver_earnings = price * 0.75
            if driver_earnings < self.constraints.minimum_driver_earnings:
                violations.append(f"Driver earnings below minimum: ${driver_earnings:.2f} < ${self.constraints.minimum_driver_earnings:.2f}")
        
        return violations
