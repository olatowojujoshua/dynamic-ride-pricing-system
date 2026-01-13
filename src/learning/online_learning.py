"""
Behavioral feedback loops and online learning for dynamic pricing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import beta
import matplotlib.pyplot as plt

from ..utils.logger import logger

class FeedbackType(Enum):
    """Types of behavioral feedback"""
    RIDE_COMPLETION = "ride_completion"
    PRICE_ACCEPTANCE = "price_acceptance"
    DRIVER_AVAILABILITY = "driver_availability"
    CANCELATION_RATE = "cancelation_rate"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    DEMAND_RESPONSE = "demand_response"

class LearningStrategy(Enum):
    """Online learning strategies"""
    STOCHASTIC_GRADIENT = "stochastic_gradient"
    BANDIT_ALGORITHM = "bandit_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"
    META_LEARNING = "meta_learning"

@dataclass
class FeedbackEvent:
    """Single feedback event from user behavior"""
    timestamp: datetime
    user_id: str
    event_type: FeedbackType
    context: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    reward: float = 0.0

@dataclass
class LearningState:
    """State of online learning system"""
    model_version: int
    samples_processed: int
    last_update: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)

class BehavioralFeedbackCollector:
    """Collect and process behavioral feedback"""
    
    def __init__(self, max_events: int = 10000):
        self.feedback_buffer = deque(maxlen=max_events)
        self.event_aggregates = defaultdict(list)
        self.feedback_patterns = {}
        
    def collect_feedback(self, event: FeedbackEvent):
        """Collect a single feedback event"""
        self.feedback_buffer.append(event)
        self.event_aggregates[event.event_type].append(event)
        
        # Update patterns
        self._update_patterns(event)
        
        logger.debug(f"Collected {event.event_type} feedback from user {event.user_id}")
    
    def _update_patterns(self, event: FeedbackEvent):
        """Update behavioral patterns based on new event"""
        event_type = event.event_type.value
        
        if event_type not in self.feedback_patterns:
            self.feedback_patterns[event_type] = {
                'total_events': 0,
                'average_reward': 0.0,
                'context_effects': defaultdict(list),
                'time_patterns': defaultdict(list)
            }
        
        pattern = self.feedback_patterns[event_type]
        pattern['total_events'] += 1
        
        # Update average reward
        pattern['average_reward'] = (
            (pattern['average_reward'] * (pattern['total_events'] - 1) + event.reward) / 
            pattern['total_events']
        )
        
        # Context effects
        for key, value in event.context.items():
            if isinstance(value, (int, float)):
                pattern['context_effects'][key].append((value, event.reward))
        
        # Time patterns
        hour = event.timestamp.hour
        pattern['time_patterns'][hour].append(event.reward)
    
    def get_feedback_summary(self, event_type: FeedbackType = None, 
                           time_window: timedelta = None) -> Dict[str, Any]:
        """Get summary of collected feedback"""
        
        events = list(self.feedback_buffer)
        
        # Filter by event type
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Filter by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            events = [e for e in events if e.timestamp >= cutoff_time]
        
        if not events:
            return {'message': 'No events found'}
        
        # Calculate summary statistics
        rewards = [e.reward for e in events]
        summary = {
            'total_events': len(events),
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_range': (min(rewards), max(rewards)),
            'event_types': defaultdict(int),
            'time_distribution': defaultdict(int)
        }
        
        # Event type distribution
        for event in events:
            summary['event_types'][event.event_type.value] += 1
            summary['time_distribution'][event.timestamp.hour] += 1
        
        return summary
    
    def analyze_price_sensitivity(self) -> Dict[str, Any]:
        """Analyze price sensitivity from feedback"""
        
        price_events = [e for e in self.feedback_buffer 
                      if e.event_type == FeedbackType.PRICE_ACCEPTANCE]
        
        if len(price_events) < 10:
            return {'message': 'Insufficient price acceptance data'}
        
        # Extract price and acceptance data
        prices = []
        acceptances = []
        contexts = []
        
        for event in price_events:
            price = event.context.get('price', 0)
            accepted = event.outcome.get('accepted', False)
            prices.append(price)
            acceptances.append(1 if accepted else 0)
            contexts.append(event.context)
        
        # Calculate price elasticity
        price_acceptance_df = pd.DataFrame({
            'price': prices,
            'acceptance': acceptances
        })
        
        # Simple elasticity calculation
        price_bins = pd.qcut(price_acceptance_df['price'], q=5, duplicates='drop')
        acceptance_by_price = price_acceptance_df.groupby(price_bins)['acceptance'].mean()
        
        elasticity = self._calculate_elasticity(acceptance_by_price)
        
        return {
            'elasticity': elasticity,
            'acceptance_by_price_bin': acceptance_by_price.to_dict(),
            'optimal_price_range': self._find_optimal_price_range(acceptance_by_price),
            'total_price_events': len(price_events)
        }
    
    def _calculate_elasticity(self, acceptance_by_price) -> float:
        """Calculate price elasticity from acceptance data"""
        
        if len(acceptance_by_price) < 2:
            return 0.0
        
        prices = [interval.mid for interval in acceptance_by_price.index]
        acceptances = acceptance_by_price.values
        
        # Calculate percentage changes
        price_changes = np.diff(prices) / prices[:-1]
        acceptance_changes = np.diff(acceptances) / acceptances[:-1]
        
        # Avoid division by zero
        valid_indices = (price_changes != 0) & (acceptance_changes != 0)
        
        if not np.any(valid_indices):
            return 0.0
        
        elasticities = acceptance_changes[valid_indices] / price_changes[valid_indices]
        return np.mean(elasticities)
    
    def _find_optimal_price_range(self, acceptance_by_price) -> Tuple[float, float]:
        """Find optimal price range based on acceptance rates"""
        
        if len(acceptance_by_price) < 2:
            return (0.0, 0.0)
        
        # Find price range with acceptance > 70%
        high_acceptance = acceptance_by_price[acceptance_by_price > 0.7]
        
        if len(high_acceptance) == 0:
            return (0.0, 0.0)
        
        prices = [interval.mid for interval in high_acceptance.index]
        return (min(prices), max(prices))

class OnlineLearningEngine:
    """Online learning engine for adaptive pricing"""
    
    def __init__(self, strategy: LearningStrategy = LearningStrategy.STOCHASTIC_GRADIENT):
        self.strategy = strategy
        self.model = None
        self.learning_state = LearningState(
            model_version=1,
            samples_processed=0,
            last_update=datetime.now()
        )
        self.performance_history = deque(maxlen=1000)
        
        # Initialize model based on strategy
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize model based on learning strategy"""
        
        if self.strategy == LearningStrategy.STOCHASTIC_GRADIENT:
            self.model = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
        elif self.strategy == LearningStrategy.ADAPTIVE_ENSEMBLE:
            self.model = AdaptiveEnsemble()
        elif self.strategy == LearningStrategy.BANDIT_ALGORITHM:
            self.model = PricingBandit()
        else:
            # Default to SGD
            self.model = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
        
        logger.info(f"Initialized online learning model with strategy: {self.strategy.value}")
    
    def update_model(self, features: np.ndarray, target: float, 
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Update model with new data point"""
        
        try:
            # Update model
            if self.strategy == LearningStrategy.STOCHASTIC_GRADIENT:
                update_result = self._update_sgd(features, target)
            elif self.strategy == LearningStrategy.ADAPTIVE_ENSEMBLE:
                update_result = self._update_ensemble(features, target, context)
            elif self.strategy == LearningStrategy.BANDIT_ALGORITHM:
                update_result = self._update_bandit(features, target, context)
            else:
                update_result = self._update_sgd(features, target)
            
            # Update learning state
            self.learning_state.samples_processed += 1
            self.learning_state.last_update = datetime.now()
            
            # Record performance
            prediction = self.predict(features.reshape(1, -1))[0]
            error = abs(prediction - target)
            self.performance_history.append({
                'timestamp': datetime.now(),
                'error': error,
                'prediction': prediction,
                'actual': target
            })
            
            # Update performance metrics
            self._update_performance_metrics()
            
            logger.debug(f"Model updated: samples={self.learning_state.samples_processed}, error={error:.4f}")
            
            return update_result
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _update_sgd(self, features: np.ndarray, target: float) -> Dict[str, Any]:
        """Update stochastic gradient descent model"""
        
        # SGD expects 2D array for features
        features_2d = features.reshape(1, -1)
        
        # Partial fit
        self.model.partial_fit(features_2d, [target])
        
        # Get prediction and error
        prediction = self.model.predict(features_2d)[0]
        error = abs(prediction - target)
        
        return {
            'strategy': 'sgd',
            'prediction': prediction,
            'error': error,
            'learning_rate': self.model.eta0
        }
    
    def _update_ensemble(self, features: np.ndarray, target: float, 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Update adaptive ensemble model"""
        
        return self.model.update(features, target, context)
    
    def _update_bandit(self, features: np.ndarray, target: float, 
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Update bandit algorithm"""
        
        return self.model.update(features, target, context)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with current model"""
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            if self.strategy == LearningStrategy.STOCHASTIC_GRADIENT:
                return self.model.predict(features)
            elif self.strategy == LearningStrategy.ADAPTIVE_ENSEMBLE:
                return self.model.predict(features)
            elif self.strategy == LearningStrategy.BANDIT_ALGORITHM:
                return self.model.predict(features)
            else:
                return self.model.predict(features)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([0.0] * len(features))
    
    def _update_performance_metrics(self):
        """Update performance metrics in learning state"""
        
        if len(self.performance_history) < 2:
            return
        
        recent_errors = [p['error'] for p in list(self.performance_history)[-100:]]
        
        self.learning_state.performance_metrics = {
            'recent_mae': np.mean(recent_errors),
            'recent_rmse': np.sqrt(np.mean([e**2 for e in recent_errors])),
            'error_trend': self._calculate_error_trend(),
            'convergence_rate': self._calculate_convergence_rate()
        }
    
    def _calculate_error_trend(self) -> float:
        """Calculate error trend (positive = increasing errors)"""
        
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_errors = [p['error'] for p in list(self.performance_history)[-10:]]
        early_errors = [p['error'] for p in list(self.performance_history)[-20:-10]]
        
        if len(early_errors) == 0:
            return 0.0
        
        recent_avg = np.mean(recent_errors)
        early_avg = np.mean(early_errors)
        
        return (recent_avg - early_avg) / early_avg if early_avg > 0 else 0.0
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate"""
        
        if len(self.performance_history) < 20:
            return 0.0
        
        errors = [p['error'] for p in list(self.performance_history)[-20:]]
        
        # Simple convergence metric: error reduction rate
        if len(errors) < 2:
            return 0.0
        
        error_reduction = (errors[0] - errors[-1]) / errors[0] if errors[0] > 0 else 0.0
        return error_reduction
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        
        return {
            'strategy': self.strategy.value,
            'learning_state': {
                'model_version': self.learning_state.model_version,
                'samples_processed': self.learning_state.samples_processed,
                'last_update': self.learning_state.last_update.isoformat(),
                'performance_metrics': self.learning_state.performance_metrics
            },
            'recent_performance': {
                'avg_error': np.mean([p['error'] for p in list(self.performance_history)[-100:]]),
                'error_std': np.std([p['error'] for p in list(self.performance_history)[-100:]]),
                'total_predictions': len(self.performance_history)
            }
        }

class AdaptiveEnsemble:
    """Adaptive ensemble for online learning"""
    
    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = [SGDRegressor(learning_rate='adaptive', eta0=0.01, random_state=42+i) 
                      for i in range(n_models)]
        self.weights = np.ones(n_models) / n_models
        self.performance_history = [[] for _ in range(n_models)]
        
    def update(self, features: np.ndarray, target: float, 
               context: Dict[str, Any]) -> Dict[str, Any]:
        """Update ensemble with new data"""
        
        features_2d = features.reshape(1, -1)
        predictions = []
        errors = []
        
        # Update each model and collect predictions
        for i, model in enumerate(self.models):
            model.partial_fit(features_2d, [target])
            pred = model.predict(features_2d)[0]
            error = abs(pred - target)
            
            predictions.append(pred)
            errors.append(error)
            self.performance_history[i].append(error)
        
        # Update weights based on recent performance
        self._update_weights()
        
        # Ensemble prediction
        ensemble_pred = np.average(predictions, weights=self.weights)
        ensemble_error = abs(ensemble_pred - target)
        
        return {
            'strategy': 'adaptive_ensemble',
            'ensemble_prediction': ensemble_pred,
            'ensemble_error': ensemble_error,
            'individual_predictions': predictions,
            'individual_errors': errors,
            'model_weights': self.weights.tolist()
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        
        predictions = []
        for model in self.models:
            pred = model.predict(features)[0]
            predictions.append(pred)
        
        return np.array([np.average(predictions, weights=self.weights)])
    
    def _update_weights(self):
        """Update model weights based on recent performance"""
        
        # Use recent performance (last 20 samples)
        recent_performances = []
        for i in range(self.n_models):
            if len(self.performance_history[i]) > 0:
                recent_errors = self.performance_history[i][-20:]
                avg_error = np.mean(recent_errors)
                recent_performances.append(1.0 / (avg_error + 1e-6))  # Inverse error
            else:
                recent_performances.append(1.0)
        
        # Normalize weights
        total_performance = sum(recent_performances)
        if total_performance > 0:
            self.weights = np.array(recent_performances) / total_performance
        else:
            self.weights = np.ones(self.n_models) / self.n_models

class PricingBandit:
    """Multi-armed bandit for pricing strategy selection"""
    
    def __init__(self, n_arms: int = 5, exploration_rate: float = 0.1):
        self.n_arms = n_arms
        self.exploration_rate = exploration_rate
        self.arm_values = np.zeros(n_arms)  # Average rewards
        self.arm_counts = np.zeros(n_arms)  # Number of pulls
        self.total_pulls = 0
        
        # Pricing strategies (different price multipliers)
        self.price_multipliers = np.linspace(0.8, 1.5, n_arms)
        
    def update(self, features: np.ndarray, target: float, 
               context: Dict[str, Any]) -> Dict[str, Any]:
        """Update bandit with new outcome"""
        
        # Select arm (pricing strategy)
        arm = self._select_arm()
        price_multiplier = self.price_multipliers[arm]
        
        # Calculate reward (could be revenue, customer satisfaction, etc.)
        reward = self._calculate_reward(target, price_multiplier, context)
        
        # Update arm statistics
        self.arm_counts[arm] += 1
        self.arm_values[arm] = ((self.arm_values[arm] * (self.arm_counts[arm] - 1) + reward) / 
                               self.arm_counts[arm])
        self.total_pulls += 1
        
        return {
            'strategy': 'bandit',
            'selected_arm': arm,
            'price_multiplier': price_multiplier,
            'reward': reward,
            'estimated_values': self.arm_values.tolist(),
            'exploration_rate': self.exploration_rate
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Select pricing strategy (arm)"""
        
        arm = self._select_arm()
        price_multiplier = self.price_multipliers[arm]
        
        return np.array([price_multiplier])
    
    def _select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy"""
        
        if np.random.random() < self.exploration_rate:
            # Exploration: random arm
            return np.random.randint(0, self.n_arms)
        else:
            # Exploitation: best arm
            return np.argmax(self.arm_values)
    
    def _calculate_reward(self, target: float, price_multiplier: float, 
                         context: Dict[str, Any]) -> float:
        """Calculate reward for pricing action"""
        
        # Simple reward: revenue * acceptance probability
        adjusted_price = target * price_multiplier
        
        # Estimate acceptance probability (decreases with price)
        acceptance_prob = max(0.1, 1.0 - (price_multiplier - 1.0) * 0.5)
        
        # Expected revenue
        reward = adjusted_price * acceptance_prob
        
        return reward

class BehavioralFeedbackLoop:
    """Main behavioral feedback loop system"""
    
    def __init__(self, learning_strategy: LearningStrategy = LearningStrategy.STOCHASTIC_GRADIENT):
        self.feedback_collector = BehavioralFeedbackCollector()
        self.learning_engine = OnlineLearningEngine(learning_strategy)
        self.feedback_processors = {}
        self.adaptation_triggers = {}
        
        # Initialize feedback processors
        self._initialize_feedback_processors()
        self._initialize_adaptation_triggers()
        
    def _initialize_feedback_processors(self):
        """Initialize processors for different feedback types"""
        
        self.feedback_processors = {
            FeedbackType.RIDE_COMPLETION: self._process_ride_completion,
            FeedbackType.PRICE_ACCEPTANCE: self._process_price_acceptance,
            FeedbackType.DRIVER_AVAILABILITY: self._process_driver_availability,
            FeedbackType.CANCELATION_RATE: self._process_cancelation_rate,
            FeedbackType.CUSTOMER_SATISFACTION: self._process_customer_satisfaction,
            FeedbackType.DEMAND_RESPONSE: self._process_demand_response
        }
    
    def _initialize_adaptation_triggers(self):
        """Initialize triggers for model adaptation"""
        
        self.adaptation_triggers = {
            'performance_degradation': self._check_performance_degradation,
            'concept_drift': self._check_concept_drift,
            'feedback_volume': self._check_feedback_volume,
            'time_based': self._check_time_based_adaptation
        }
    
    def process_feedback(self, event: FeedbackEvent) -> Dict[str, Any]:
        """Process a single feedback event"""
        
        # Collect feedback
        self.feedback_collector.collect_feedback(event)
        
        # Process based on event type
        processor = self.feedback_processors.get(event.event_type)
        if processor:
            processing_result = processor(event)
        else:
            processing_result = {'status': 'no_processor'}
        
        # Check for adaptation triggers
        adaptation_needed = self._check_adaptation_triggers()
        
        if adaptation_needed:
            adaptation_result = self._trigger_adaptation()
        else:
            adaptation_result = {'status': 'no_adaptation_needed'}
        
        return {
            'feedback_processed': True,
            'processing_result': processing_result,
            'adaptation_triggered': adaptation_needed,
            'adaptation_result': adaptation_result
        }
    
    def _process_ride_completion(self, event: FeedbackEvent) -> Dict[str, Any]:
        """Process ride completion feedback"""
        
        # Extract features and outcome
        features = self._extract_features_from_context(event.context)
        actual_price = event.outcome.get('final_price', 0)
        customer_rating = event.outcome.get('customer_rating', 0)
        
        # Calculate reward based on customer satisfaction
        reward = customer_rating / 5.0  # Normalize to 0-1
        
        # Update learning model
        update_result = self.learning_engine.update_model(features, actual_price, event.context)
        
        return {
            'event_type': 'ride_completion',
            'reward': reward,
            'update_result': update_result
        }
    
    def _process_price_acceptance(self, event: FeedbackEvent) -> Dict[str, Any]:
        """Process price acceptance feedback"""
        
        features = self._extract_features_from_context(event.context)
        offered_price = event.context.get('offered_price', 0)
        accepted = event.outcome.get('accepted', False)
        
        # Reward: 1 for acceptance, 0 for rejection
        reward = 1.0 if accepted else 0.0
        
        # Update model with acceptance outcome
        update_result = self.learning_engine.update_model(features, reward, event.context)
        
        return {
            'event_type': 'price_acceptance',
            'accepted': accepted,
            'reward': reward,
            'update_result': update_result
        }
    
    def _process_driver_availability(self, event: FeedbackEvent) -> Dict[str, Any]:
        """Process driver availability feedback"""
        
        features = self._extract_features_from_context(event.context)
        surge_multiplier = event.context.get('surge_multiplier', 1.0)
        drivers_available = event.outcome.get('drivers_available', 0)
        drivers_needed = event.context.get('drivers_needed', 1)
        
        # Reward based on supply-demand balance
        supply_ratio = min(1.0, drivers_available / drivers_needed)
        reward = supply_ratio
        
        update_result = self.learning_engine.update_model(features, reward, event.context)
        
        return {
            'event_type': 'driver_availability',
            'supply_ratio': supply_ratio,
            'reward': reward,
            'update_result': update_result
        }
    
    def _process_cancelation_rate(self, event: FeedbackEvent) -> Dict[str, Any]:
        """Process cancelation rate feedback"""
        
        features = self._extract_features_from_context(event.context)
        price = event.context.get('price', 0)
        canceled = event.outcome.get('canceled', False)
        
        # Negative reward for cancelations
        reward = 0.0 if canceled else 1.0
        
        update_result = self.learning_engine.update_model(features, reward, event.context)
        
        return {
            'event_type': 'cancelation_rate',
            'canceled': canceled,
            'reward': reward,
            'update_result': update_result
        }
    
    def _process_customer_satisfaction(self, event: FeedbackEvent) -> Dict[str, Any]:
        """Process customer satisfaction feedback"""
        
        features = self._extract_features_from_context(event.context)
        satisfaction_score = event.outcome.get('satisfaction_score', 0)
        
        # Normalize satisfaction score
        reward = satisfaction_score / 5.0
        
        update_result = self.learning_engine.update_model(features, reward, event.context)
        
        return {
            'event_type': 'customer_satisfaction',
            'satisfaction_score': satisfaction_score,
            'reward': reward,
            'update_result': update_result
        }
    
    def _process_demand_response(self, event: FeedbackEvent) -> Dict[str, Any]:
        """Process demand response feedback"""
        
        features = self._extract_features_from_context(event.context)
        price_change = event.context.get('price_change', 0)
        demand_change = event.outcome.get('demand_change', 0)
        
        # Reward based on demand elasticity
        if price_change != 0:
            elasticity = demand_change / price_change
            reward = max(0.0, 1.0 - abs(elasticity))  # Prefer moderate elasticity
        else:
            reward = 0.5
        
        update_result = self.learning_engine.update_model(features, reward, event.context)
        
        return {
            'event_type': 'demand_response',
            'elasticity': elasticity if price_change != 0 else 0,
            'reward': reward,
            'update_result': update_result
        }
    
    def _extract_features_from_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from context"""
        
        feature_mapping = {
            'number_of_riders': 0,
            'number_of_drivers': 1,
            'location_category_urban': 2,
            'location_category_suburban': 3,
            'location_category_rural': 4,
            'customer_loyalty_silver': 5,
            'customer_loyalty_gold': 6,
            'customer_loyalty_platinum': 7,
            'time_of_booking_morning': 8,
            'time_of_booking_afternoon': 9,
            'time_of_booking_evening': 10,
            'time_of_booking_night': 11,
            'vehicle_type_economy': 12,
            'vehicle_type_premium': 13,
            'vehicle_type_luxury': 14,
            'expected_duration': 15
        }
        
        features = np.zeros(len(feature_mapping))
        
        # Map context to features
        for key, value in context.items():
            if key in ['number_of_riders', 'number_of_drivers', 'expected_duration']:
                if key in feature_mapping:
                    features[feature_mapping[key]] = float(value)
            elif key == 'location_category':
                location_key = f'location_category_{value.lower()}'
                if location_key in feature_mapping:
                    features[feature_mapping[location_key]] = 1.0
            elif key == 'customer_loyalty_status':
                loyalty_key = f'customer_loyalty_{value.lower()}'
                if loyalty_key in feature_mapping:
                    features[feature_mapping[loyalty_key]] = 1.0
            elif key == 'time_of_booking':
                time_key = f'time_of_booking_{value.lower()}'
                if time_key in feature_mapping:
                    features[feature_mapping[time_key]] = 1.0
            elif key == 'vehicle_type':
                vehicle_key = f'vehicle_type_{value.lower()}'
                if vehicle_key in feature_mapping:
                    features[feature_mapping[vehicle_key]] = 1.0
        
        return features
    
    def _check_adaptation_triggers(self) -> bool:
        """Check if any adaptation triggers are activated"""
        
        for trigger_name, trigger_func in self.adaptation_triggers.items():
            if trigger_func():
                logger.info(f"Adaptation trigger activated: {trigger_name}")
                return True
        
        return False
    
    def _check_performance_degradation(self) -> bool:
        """Check for performance degradation"""
        
        if len(self.learning_engine.performance_history) < 50:
            return False
        
        recent_errors = [p['error'] for p in list(self.learning_engine.performance_history)[-20:]]
        early_errors = [p['error'] for p in list(self.learning_engine.performance_history)[-50:-30]]
        
        recent_avg = np.mean(recent_errors)
        early_avg = np.mean(early_errors)
        
        # Trigger if recent error is 20% higher than early error
        return recent_avg > early_avg * 1.2
    
    def _check_concept_drift(self) -> bool:
        """Check for concept drift"""
        
        # Simple drift detection based on error pattern changes
        if len(self.learning_engine.performance_history) < 100:
            return False
        
        recent_errors = [p['error'] for p in list(self.learning_engine.performance_history)[-50:]]
        
        # Check for sudden increase in error variance
        error_variance = np.var(recent_errors)
        mean_error = np.mean(recent_errors)
        
        # High variance relative to mean might indicate drift
        return error_variance > mean_error
    
    def _check_feedback_volume(self) -> bool:
        """Check if feedback volume triggers adaptation"""
        
        # Trigger if we have enough new feedback
        recent_feedback = len([e for e in self.feedback_collector.feedback_buffer 
                              if e.timestamp > datetime.now() - timedelta(hours=1)])
        
        return recent_feedback >= 100  # Adapt every 100 feedback events
    
    def _check_time_based_adaptation(self) -> bool:
        """Check for time-based adaptation"""
        
        # Adapt every 6 hours
        time_since_last = datetime.now() - self.learning_engine.learning_state.last_update
        return time_since_last >= timedelta(hours=6)
    
    def _trigger_adaptation(self) -> Dict[str, Any]:
        """Trigger model adaptation"""
        
        logger.info("Triggering model adaptation")
        
        # Update learning state
        self.learning_engine.learning_state.model_version += 1
        
        # Could implement more sophisticated adaptation here
        # For now, just log the adaptation
        adaptation_record = {
            'timestamp': datetime.now(),
            'model_version': self.learning_engine.learning_state.model_version,
            'samples_processed': self.learning_engine.learning_state.samples_processed,
            'trigger_reason': 'automatic'
        }
        
        self.learning_engine.learning_state.adaptation_history.append(adaptation_record)
        
        return {
            'status': 'adaptation_triggered',
            'new_model_version': self.learning_engine.learning_state.model_version,
            'adaptation_record': adaptation_record
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'feedback_collector': {
                'total_events': len(self.feedback_collector.feedback_buffer),
                'event_types': {k.value: len(v) for k, v in self.feedback_collector.event_aggregates.items()},
                'feedback_summary': self.feedback_collector.get_feedback_summary()
            },
            'learning_engine': self.learning_engine.get_learning_summary(),
            'price_sensitivity': self.feedback_collector.analyze_price_sensitivity(),
            'recent_adaptations': self.learning_engine.learning_state.adaptation_history[-5:]
        }
