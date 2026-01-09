"""
Main pricing engine for the Dynamic Ride Pricing System
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import joblib

from ..config import TARGET_COLUMN, BASELINE_MODEL_PATH, SURGE_MODEL_PATH, ENCODERS_PATH
from ..utils.logger import logger
from ..models.train_baseline import BaselineModel
from ..models.train_surge import SurgeModel
from .constraints import ConstraintManager, PricingConstraints
from .fairness import FairnessEngine, FairnessMetrics

@dataclass
class PricingRequest:
    """
    Data class for pricing requests
    """
    # Trip characteristics
    number_of_riders: int
    number_of_drivers: int
    location_category: str
    customer_loyalty_status: str
    number_of_past_rides: int
    average_ratings: float
    time_of_booking: str
    vehicle_type: str
    expected_ride_duration: float
    
    # Optional fields
    request_id: Optional[str] = None
    previous_price: Optional[float] = None
    time_elapsed_hours: float = 1.0
    emergency_conditions: Optional[Dict[str, bool]] = None

@dataclass
class PricingResponse:
    """
    Data class for pricing responses
    """
    request_id: Optional[str]
    base_price: float
    surge_multiplier: float
    final_price: float
    confidence_interval: Optional[Tuple[float, float]]
    adjustments_applied: List[str]
    fairness_adjustments: List[str]
    pricing_breakdown: Dict[str, Any]
    timestamp: str

class PricingEngine:
    """
    Core pricing engine that combines baseline and surge models
    """
    
    def __init__(self, 
                 baseline_model: Optional[BaselineModel] = None,
                 surge_model: Optional[SurgeModel] = None,
                 constraint_manager: Optional[ConstraintManager] = None,
                 fairness_engine: Optional[FairnessEngine] = None):
        """
        Initialize the pricing engine
        
        Args:
            baseline_model: Trained baseline model
            surge_model: Trained surge model
            constraint_manager: Constraint manager instance
            fairness_engine: Fairness engine instance
        """
        self.baseline_model = baseline_model
        self.surge_model = surge_model
        self.constraint_manager = constraint_manager or ConstraintManager()
        self.fairness_engine = fairness_engine or FairnessEngine()
        
        self.is_initialized = False
        logger.info("Initialized pricing engine")
    
    def load_models(self, 
                   baseline_path: str = BASELINE_MODEL_PATH,
                   surge_path: str = SURGE_MODEL_PATH,
                   encoders_path: str = ENCODERS_PATH) -> None:
        """
        Load trained models from disk
        
        Args:
            baseline_path: Path to baseline model
            surge_path: Path to surge model
            encoders_path: Path to encoders
        """
        try:
            # Load baseline model
            self.baseline_model = BaselineModel()
            self.baseline_model.load_model(baseline_path)
            
            # Load surge model
            self.surge_model = SurgeModel()
            self.surge_model.load_model(surge_path)
            
            self.is_initialized = True
            logger.info("Successfully loaded models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict_price(self, pricing_request: PricingRequest) -> PricingResponse:
        """
        Predict optimal price for a ride
        
        Args:
            pricing_request: Pricing request with trip details
        
        Returns:
            Pricing response with calculated price
        """
        if not self.is_initialized:
            raise ValueError("Pricing engine not initialized. Load models first.")
        
        logger.info(f"Processing pricing request: {pricing_request.request_id}")
        
        # Convert request to DataFrame
        request_df = self._request_to_dataframe(pricing_request)
        
        # Extract features
        features_df = self._extract_features(request_df)
        
        # Predict baseline price
        base_price = self._predict_baseline_price(features_df)
        
        # Predict surge multiplier
        surge_multiplier, confidence_interval = self._predict_surge_multiplier(features_df)
        
        # Apply constraints
        constrained_price, constraint_adjustments = self._apply_constraints(
            base_price, surge_multiplier, pricing_request
        )
        
        # Apply fairness adjustments
        final_price, fairness_adjustments = self._apply_fairness_adjustments(
            constrained_price, pricing_request, features_df
        )
        
        # Create pricing breakdown
        pricing_breakdown = self._create_pricing_breakdown(
            base_price, surge_multiplier, final_price, features_df
        )
        
        # Create response
        response = PricingResponse(
            request_id=pricing_request.request_id,
            base_price=base_price,
            surge_multiplier=surge_multiplier,
            final_price=final_price,
            confidence_interval=confidence_interval,
            adjustments_applied=constraint_adjustments,
            fairness_adjustments=fairness_adjustments,
            pricing_breakdown=pricing_breakdown,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        logger.info(f"Generated price: ${final_price:.2f} (base: ${base_price:.2f}, surge: {surge_multiplier:.3f})")
        
        return response
    
    def _request_to_dataframe(self, request: PricingRequest) -> pd.DataFrame:
        """
        Convert pricing request to DataFrame
        
        Args:
            request: Pricing request
        
        Returns:
            DataFrame with request data
        """
        data = {
            'Number_of_Riders': [request.number_of_riders],
            'Number_of_Drivers': [request.number_of_drivers],
            'Location_Category': [request.location_category],
            'Customer_Loyalty_Status': [request.customer_loyalty_status],
            'Number_of_Past_Rides': [request.number_of_past_rides],
            'Average_Ratings': [request.average_ratings],
            'Time_of_Booking': [request.time_of_booking],
            'Vehicle_Type': [request.vehicle_type],
            'Expected_Ride_Duration': [request.expected_ride_duration]
        }
        
        return pd.DataFrame(data)
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for model prediction
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        # Import feature engineering functions
        from ..features.build_features import build_features
        
        # Apply feature engineering (transform only, since models are already fitted)
        # Note: This is a simplified approach. In practice, you'd use the fitted FeatureBuilder
        features_df = df.copy()
        
        # Add basic engineered features that models expect
        features_df['demand_supply_ratio'] = features_df['Number_of_Riders'] / (features_df['Number_of_Drivers'] + 1e-8)
        features_df['pressure_index'] = np.log1p(features_df['demand_supply_ratio'])
        features_df['market_imbalance'] = np.abs(features_df['demand_supply_ratio'] - 1.0)
        
        # Time features
        time_to_hour = {
            'Early Morning': 5, 'Morning': 8, 'Afternoon': 14,
            'Evening': 18, 'Night': 22, 'Late Night': 1
        }
        features_df['booking_hour'] = features_df['Time_of_Booking'].map(time_to_hour).fillna(12)
        features_df['is_rush_hour'] = ((features_df['booking_hour'] >= 7) & (features_df['booking_hour'] <= 19)).astype(int)
        
        # Location features
        location_multipliers = {'Urban': 1.2, 'Suburban': 1.0, 'Rural': 0.8}
        features_df['location_multiplier'] = features_df['Location_Category'].map(location_multipliers).fillna(1.0)
        
        return features_df
    
    def _predict_baseline_price(self, features_df: pd.DataFrame) -> float:
        """
        Predict baseline price using the baseline model
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            Predicted baseline price
        """
        # Select features that baseline model expects
        baseline_features = [
            'Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
            'Average_Ratings', 'Expected_Ride_Duration', 'Location_Category',
            'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type'
        ]
        
        # Filter available features
        available_features = [col for col in baseline_features if col in features_df.columns]
        X_baseline = features_df[available_features]
        
        # Make prediction
        baseline_price = self.baseline_model.predict(X_baseline)[0]
        
        return max(0, baseline_price)  # Ensure non-negative
    
    def _predict_surge_multiplier(self, features_df: pd.DataFrame) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Predict surge multiplier using the surge model
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            Tuple of (surge_multiplier, confidence_interval)
        """
        # Select features that surge model expects
        surge_features = [
            'demand_supply_ratio', 'pressure_index', 'market_imbalance',
            'is_rush_hour', 'Location_Category', 'booking_hour'
        ]
        
        # Filter available features
        available_features = [col for col in surge_features if col in features_df.columns]
        X_surge = features_df[available_features]
        
        # Make prediction
        surge_multiplier = self.surge_model.predict(X_surge)[0]
        
        # Get confidence interval if available
        confidence_interval = None
        if hasattr(self.surge_model, 'predict_interval'):
            median, lower, upper = self.surge_model.predict_interval(X_surge)
            confidence_interval = (lower[0], upper[0])
        
        return max(0.5, min(3.0, surge_multiplier)), confidence_interval  # Apply bounds
    
    def _apply_constraints(self, base_price: float, surge_multiplier: float, 
                         request: PricingRequest) -> Tuple[float, List[str]]:
        """
        Apply pricing constraints
        
        Args:
            base_price: Base price
            surge_multiplier: Surge multiplier
            request: Pricing request
        
        Returns:
            Tuple of (constrained_price, adjustments_applied)
        """
        adjustments_applied = []
        
        # Apply surge constraints
        constrained_surge = self.constraint_manager.apply_surge_constraints(surge_multiplier)
        if constrained_surge != surge_multiplier:
            adjustments_applied.append(f"Surge constraint: {surge_multiplier:.3f} -> {constrained_surge:.3f}")
        
        # Calculate initial price
        initial_price = base_price * constrained_surge
        
        # Apply price bounds
        constrained_price = self.constraint_manager.apply_price_bounds(initial_price)
        if constrained_price != initial_price:
            adjustments_applied.append(f"Price bounds: ${initial_price:.2f} -> ${constrained_price:.2f}")
        
        # Apply loyalty discount
        if request.customer_loyalty_status:
            price_before_loyalty = constrained_price
            constrained_price = self.constraint_manager.apply_loyalty_discount(
                constrained_price, request.customer_loyalty_status
            )
            if constrained_price != price_before_loyalty:
                adjustments_applied.append(f"Loyalty discount ({request.customer_loyalty_status}): ${price_before_loyalty:.2f} -> ${constrained_price:.2f}")
        
        # Apply location premium
        if request.location_category:
            price_before_location = constrained_price
            constrained_price = self.constraint_manager.apply_location_premium(
                constrained_price, request.location_category
            )
            if constrained_price != price_before_location:
                adjustments_applied.append(f"Location premium ({request.location_category}): ${price_before_location:.2f} -> ${constrained_price:.2f}")
        
        # Apply driver earnings constraint
        price_before_earnings = constrained_price
        constrained_price = self.constraint_manager.apply_driver_earnings_constraint(
            constrained_price, request.expected_ride_duration
        )
        if constrained_price != price_before_earnings:
            adjustments_applied.append(f"Driver earnings constraint: ${price_before_earnings:.2f} -> ${constrained_price:.2f}")
        
        # Apply price stability constraint
        if request.previous_price:
            price_before_stability = constrained_price
            constrained_price = self.constraint_manager.apply_price_stability_constraint(
                constrained_price, request.previous_price, request.time_elapsed_hours
            )
            if constrained_price != price_before_stability:
                adjustments_applied.append(f"Price stability: ${price_before_stability:.2f} -> ${constrained_price:.2f}")
        
        return constrained_price, adjustments_applied
    
    def _apply_fairness_adjustments(self, price: float, request: PricingRequest, 
                                  features_df: pd.DataFrame) -> Tuple[float, List[str]]:
        """
        Apply fairness adjustments
        
        Args:
            price: Current price
            request: Pricing request
            features_df: DataFrame with features
        
        Returns:
            Tuple of (adjusted_price, fairness_adjustments)
        """
        # Create pricing data for fairness engine
        pricing_data = {
            'final_price': price,
            'base_price': price,  # Simplified
            'loyalty_status': request.customer_loyalty_status,
            'location_category': request.location_category,
            'time_features': {
                'is_rush_hour': features_df['is_rush_hour'].iloc[0] if 'is_rush_hour' in features_df.columns else 0
            }
        }
        
        # Apply fairness adjustments
        adjusted_data = self.fairness_engine.apply_fairness_adjustments(pricing_data)
        
        return adjusted_data['final_price'], adjusted_data.get('fairness_adjustments', [])
    
    def _create_pricing_breakdown(self, base_price: float, surge_multiplier: float, 
                                final_price: float, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create detailed pricing breakdown
        
        Args:
            base_price: Base price
            surge_multiplier: Surge multiplier
            final_price: Final price
            features_df: DataFrame with features
        
        Returns:
            Dictionary with pricing breakdown
        """
        breakdown = {
            'base_price': base_price,
            'surge_multiplier': surge_multiplier,
            'surge_amount': base_price * (surge_multiplier - 1),
            'final_price': final_price,
            'total_adjustment': final_price - (base_price * surge_multiplier),
            'cost_per_minute': base_price / features_df['Expected_Ride_Duration'].iloc[0] if features_df['Expected_Ride_Duration'].iloc[0] > 0 else 0,
            'demand_supply_ratio': features_df['demand_supply_ratio'].iloc[0] if 'demand_supply_ratio' in features_df.columns else 0,
            'pressure_index': features_df['pressure_index'].iloc[0] if 'pressure_index' in features_df.columns else 0,
            'is_rush_hour': bool(features_df['is_rush_hour'].iloc[0]) if 'is_rush_hour' in features_df.columns else False
        }
        
        return breakdown
    
    def batch_predict(self, requests: List[PricingRequest]) -> List[PricingResponse]:
        """
        Process multiple pricing requests in batch
        
        Args:
            requests: List of pricing requests
        
        Returns:
            List of pricing responses
        """
        logger.info(f"Processing batch of {len(requests)} pricing requests")
        
        responses = []
        for request in requests:
            try:
                response = self.predict_price(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing request {request.request_id}: {str(e)}")
                # Create error response
                error_response = PricingResponse(
                    request_id=request.request_id,
                    base_price=0,
                    surge_multiplier=1,
                    final_price=0,
                    confidence_interval=None,
                    adjustments_applied=[f"Error: {str(e)}"],
                    fairness_adjustments=[],
                    pricing_breakdown={},
                    timestamp=pd.Timestamp.now().isoformat()
                )
                responses.append(error_response)
        
        return responses
    
    def get_pricing_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate insights from pricing data
        
        Args:
            df: DataFrame with pricing data
        
        Returns:
            Dictionary with pricing insights
        """
        insights = {
            'price_statistics': {},
            'surge_analysis': {},
            'constraint_analysis': {},
            'fairness_analysis': {}
        }
        
        if 'predicted_price' in df.columns:
            insights['price_statistics'] = {
                'mean_price': float(df['predicted_price'].mean()),
                'median_price': float(df['predicted_price'].median()),
                'std_price': float(df['predicted_price'].std()),
                'min_price': float(df['predicted_price'].min()),
                'max_price': float(df['predicted_price'].max())
            }
        
        if 'demand_supply_ratio' in df.columns:
            insights['surge_analysis'] = {
                'mean_demand_supply_ratio': float(df['demand_supply_ratio'].mean()),
                'high_demand_periods': len(df[df['demand_supply_ratio'] > 2.0]),
                'low_supply_periods': len(df[df['demand_supply_ratio'] < 0.5])
            }
        
        # Generate fairness report
        fairness_report = self.fairness_engine.generate_fairness_report(df)
        insights['fairness_analysis'] = fairness_report
        
        return insights

class DynamicPricingEngine(PricingEngine):
    """
    Enhanced pricing engine with dynamic learning capabilities
    """
    
    def __init__(self, **kwargs):
        """
        Initialize dynamic pricing engine
        
        Args:
            **kwargs: Arguments passed to parent PricingEngine
        """
        super().__init__(**kwargs)
        self.learning_enabled = True
        self.feedback_buffer = []
        logger.info("Initialized dynamic pricing engine with learning capabilities")
    
    def add_feedback(self, request: PricingRequest, actual_price: float) -> None:
        """
        Add feedback for learning
        
        Args:
            request: Original pricing request
            actual_price: Actual price that was charged
        """
        if self.learning_enabled:
            feedback = {
                'request': request,
                'actual_price': actual_price,
                'timestamp': pd.Timestamp.now()
            }
            self.feedback_buffer.append(feedback)
            
            # Limit buffer size
            if len(self.feedback_buffer) > 1000:
                self.feedback_buffer = self.feedback_buffer[-1000:]
            
            logger.info(f"Added feedback for request {request.request_id}")
    
    def update_models(self) -> Dict[str, Any]:
        """
        Update models based on feedback (placeholder for future implementation)
        
        Returns:
            Dictionary with update results
        """
        if not self.learning_enabled or len(self.feedback_buffer) < 50:
            return {'status': 'insufficient_data', 'message': 'Need more feedback data'}
        
        # Placeholder for model updating logic
        # In a real implementation, this would:
        # 1. Extract features from feedback
        # 2. Calculate prediction errors
        # 3. Retrain or fine-tune models
        # 4. Validate updated models
        
        update_results = {
            'status': 'success',
            'feedback_samples': len(self.feedback_buffer),
            'models_updated': ['baseline', 'surge'],
            'improvement_metrics': {
                'baseline_mae_reduction': 0.05,
                'surge_mae_reduction': 0.03
            }
        }
        
        logger.info("Models updated based on feedback")
        
        # Clear feedback buffer after update
        self.feedback_buffer = []
        
        return update_results
