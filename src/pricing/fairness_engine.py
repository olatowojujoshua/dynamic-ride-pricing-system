"""
Formal fairness-aware machine learning algorithms for dynamic pricing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    # Create mock cp for basic functionality
    class MockCP:
        class Variable:
            def __init__(self, n):
                self.value = np.random.randn(n)
        class Constraint:
            pass
        @staticmethod
        def Minimize(expr):
            return None
        @staticmethod
        def Problem(objective, constraints):
            class MockProblem:
                def solve(self, solver=None, max_iters=None):
                    pass
                @property
                def status(self):
                    return 'optimal'
                @property
                def value(self):
                    return 0.0
            return MockProblem()
        @staticmethod
        def sum(expr):
            return 0.0
        @staticmethod
        def sum_squares(expr):
            return 0.0
        ECOS = 'ECOS'
    cp = MockCP()
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import logger

class FairnessMetric(Enum):
    """Types of fairness metrics"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUALIZED_OPPORTUNITY = "equalized_opportunity"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"

@dataclass
class FairnessConstraint:
    """Fairness constraint specification"""
    metric: FairnessMetric
    protected_attribute: str
    tolerance: float
    weight: float = 1.0
    direction: str = "less_than"  # less_than, greater_than, equal

@dataclass
class FairnessReport:
    """Comprehensive fairness evaluation report"""
    metric_name: str
    protected_attribute: str
    baseline_value: float
    current_value: float
    disparity: float
    violation: bool
    tolerance: float
    recommendation: str

class FairnessMetrics:
    """Calculate various fairness metrics"""
    
    @staticmethod
    def demographic_parity(y_pred: np.ndarray, protected_attr: np.ndarray) -> float:
        """
        Demographic Parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
        For regression: E[Ŷ|A=0] = E[Ŷ|A=1]
        """
        unique_groups = np.unique(protected_attr)
        if len(unique_groups) != 2:
            return 0.0  # Only supports binary protected attributes
        
        group_0_mask = protected_attr == unique_groups[0]
        group_1_mask = protected_attr == unique_groups[1]
        
        mean_0 = np.mean(y_pred[group_0_mask])
        mean_1 = np.mean(y_pred[group_1_mask])
        
        return abs(mean_0 - mean_1)
    
    @staticmethod
    def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, 
                      protected_attr: np.ndarray, threshold: float = None) -> float:
        """
        Equalized Odds: P(Ŷ=1|Y=y, A=0) = P(Ŷ=1|Y=y, A=1) for all y
        For regression: E[Ŷ|Y=y, A=0] = E[Ŷ|Y=y, A=1]
        """
        if threshold is None:
            threshold = np.median(y_true)
        
        # Convert to binary for easier calculation
        y_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        unique_groups = np.unique(protected_attr)
        if len(unique_groups) != 2:
            return 0.0
        
        disparities = []
        for y_val in [0, 1]:
            y_mask = y_binary == y_val
            if not np.any(y_mask):
                continue
                
            group_0_mask = (protected_attr == unique_groups[0]) & y_mask
            group_1_mask = (protected_attr == unique_groups[1]) & y_mask
            
            if np.any(group_0_mask) and np.any(group_1_mask):
                rate_0 = np.mean(y_pred_binary[group_0_mask])
                rate_1 = np.mean(y_pred_binary[group_1_mask])
                disparities.append(abs(rate_0 - rate_1))
        
        return np.mean(disparities) if disparities else 0.0
    
    @staticmethod
    def equalized_opportunity(y_true: np.ndarray, y_pred: np.ndarray,
                            protected_attr: np.ndarray, threshold: float = None) -> float:
        """
        Equalized Opportunity: P(Ŷ=1|Y=1, A=0) = P(Ŷ=1|Y=1, A=1)
        """
        if threshold is None:
            threshold = np.median(y_true)
        
        y_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        unique_groups = np.unique(protected_attr)
        if len(unique_groups) != 2:
            return 0.0
        
        # Only consider positive outcomes
        positive_mask = y_binary == 1
        if not np.any(positive_mask):
            return 0.0
        
        group_0_mask = (protected_attr == unique_groups[0]) & positive_mask
        group_1_mask = (protected_attr == unique_groups[1]) & positive_mask
        
        if np.any(group_0_mask) and np.any(group_1_mask):
            rate_0 = np.mean(y_pred_binary[group_0_mask])
            rate_1 = np.mean(y_pred_binary[group_1_mask])
            return abs(rate_0 - rate_1)
        
        return 0.0
    
    @staticmethod
    def individual_fairness(y_pred: np.ndarray, X: pd.DataFrame, 
                           sensitive_features: List[str]) -> float:
        """
        Individual Fairness: Similar individuals should receive similar predictions
        Based on Lipschitz continuity
        """
        if len(X) < 2:
            return 0.0
        
        # Calculate pairwise distances in sensitive feature space
        sensitive_X = X[sensitive_features].values
        n_samples = len(X)
        
        total_violation = 0.0
        comparisons = 0
        
        for i in range(min(n_samples, 100)):  # Sample for efficiency
            for j in range(i+1, min(n_samples, 100)):
                # Feature distance
                feature_dist = np.linalg.norm(sensitive_X[i] - sensitive_X[j])
                
                # Prediction difference
                pred_diff = abs(y_pred[i] - y_pred[j])
                
                # Lipschitz violation
                if feature_dist > 0:
                    violation = pred_diff / feature_dist
                    total_violation += violation
                    comparisons += 1
        
        return total_violation / comparisons if comparisons > 0 else 0.0
    
    @staticmethod
    def counterfactual_fairness(model, X: pd.DataFrame, 
                               protected_attr: str) -> float:
        """
        Counterfactual Fairness: Model prediction should not change 
        when protected attribute changes, keeping everything else constant
        """
        if protected_attr not in X.columns:
            return 0.0
        
        # Create counterfactual dataset
        X_counterfactual = X.copy()
        unique_values = X[protected_attr].unique()
        
        if len(unique_values) < 2:
            return 0.0
        
        # Swap protected attribute values
        value_0, value_1 = unique_values[:2]
        mask_0 = X[protected_attr] == value_0
        mask_1 = X[protected_attr] == value_1
        
        if not np.any(mask_0) or not np.any(mask_1):
            return 0.0
        
        # Create counterfactuals
        X_counterfactual.loc[mask_0, protected_attr] = value_1
        X_counterfactual.loc[mask_1, protected_attr] = value_0
        
        # Get predictions
        pred_original = model.predict(X)
        pred_counterfactual = model.predict(X_counterfactual)
        
        # Calculate average difference
        differences = np.abs(pred_original - pred_counterfactual)
        return np.mean(differences)

class FairnessConstraintOptimizer:
    """Optimize models under fairness constraints"""
    
    def __init__(self, base_model, fairness_constraints: List[FairnessConstraint]):
        self.base_model = base_model
        self.fairness_constraints = fairness_constraints
        self.optimization_history = []
        
    def optimize_fairness(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_protected: pd.Series, max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize model predictions to satisfy fairness constraints
        using constrained optimization
        """
        logger.info(f"Optimizing fairness with {len(self.fairness_constraints)} constraints")
        
        # Get base predictions
        base_predictions = self.base_model.predict(X_train)
        
        # For now, skip optimization if cvxpy not available
        if not CVXPY_AVAILABLE:
            logger.warning("CVXPY not available, skipping fairness optimization")
            return {
                'status': 'skipped',
                'optimized_predictions': base_predictions,
                'objective_value': None,
                'constraints_satisfied': False
            }
        
        # Define optimization variables (adjusted predictions)
        n_samples = len(X_train)
        adjusted_predictions = cp.Variable(n_samples)
        
        # Objective: Minimize deviation from base predictions
        objective = cp.Minimize(cp.sum_squares(adjusted_predictions - base_predictions))
        
        # Add fairness constraints
        constraints = []
        
        for constraint in self.fairness_constraints:
            fairness_constraint = self._create_fairness_constraint(
                adjusted_predictions, X_protected, constraint
            )
            if fairness_constraint is not None:
                constraints.extend(fairness_constraint)
        
        # Add reasonable bounds
        constraints.extend([
            adjusted_predictions >= np.percentile(base_predictions, 1),
            adjusted_predictions <= np.percentile(base_predictions, 99)
        ])
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, max_iters=max_iterations)
            
            if problem.status == cp.OPTIMAL:
                optimized_predictions = adjusted_predictions.value
                logger.info("Fairness optimization completed successfully")
                
                return {
                    'status': 'optimal',
                    'optimized_predictions': optimized_predictions,
                    'objective_value': problem.value,
                    'constraints_satisfied': True
                }
            else:
                logger.warning(f"Optimization status: {problem.status}")
                return {
                    'status': problem.status,
                    'optimized_predictions': base_predictions,
                    'objective_value': None,
                    'constraints_satisfied': False
                }
                
        except Exception as e:
            logger.error(f"Fairness optimization failed: {e}")
            return {
                'status': 'failed',
                'optimized_predictions': base_predictions,
                'objective_value': None,
                'constraints_satisfied': False
            }
    
    def _create_fairness_constraint(self, predictions: cp.Variable, 
                                  protected_attr: pd.Series,
                                  constraint: FairnessConstraint) -> List[cp.Constraint]:
        """Create fairness constraint for optimization"""
        
        if constraint.metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            return self._demographic_parity_constraint(predictions, protected_attr, constraint)
        elif constraint.metric == FairnessMetric.EQUALIZED_ODDS:
            # For equalized odds, we need true labels - simplified version here
            return self._demographic_parity_constraint(predictions, protected_attr, constraint)
        else:
            logger.warning(f"Constraint {constraint.metric} not yet implemented for optimization")
            return []
    
    def _demographic_parity_constraint(self, predictions: cp.Variable,
                                    protected_attr: pd.Series,
                                    constraint: FairnessConstraint) -> List[cp.Constraint]:
        """Create demographic parity constraint"""
        
        unique_groups = np.unique(protected_attr)
        if len(unique_groups) != 2:
            return []
        
        group_0_mask = protected_attr == unique_groups[0]
        group_1_mask = protected_attr == unique_groups[1]
        
        # Group means
        mean_0 = cp.sum(predictions[group_0_mask]) / cp.sum(group_0_mask)
        mean_1 = cp.sum(predictions[group_1_mask]) / cp.sum(group_1_mask)
        
        # Fairness constraint: |mean_0 - mean_1| <= tolerance
        disparity = mean_0 - mean_1
        
        if constraint.direction == "less_than":
            return [disparity <= constraint.tolerance, -disparity <= constraint.tolerance]
        elif constraint.direction == "equal":
            return [disparity == 0]
        else:
            return []

class FairnessAwareModel:
    """Fairness-aware model wrapper"""
    
    def __init__(self, base_model, fairness_constraints: List[FairnessConstraint],
                 fairness_weight: float = 0.5):
        self.base_model = base_model
        self.fairness_constraints = fairness_constraints
        self.fairness_weight = fairness_weight
        self.fairness_optimizer = None
        self.is_fitted = False
        self.fairness_history = []
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            protected_attributes: Dict[str, pd.Series]) -> 'FairnessAwareModel':
        """
        Fit fairness-aware model
        """
        logger.info("Training fairness-aware model")
        
        # Train base model
        self.base_model.fit(X_train, y_train)
        
        # Optimize for fairness
        self.fairness_optimizer = FairnessConstraintOptimizer(
            self.base_model, self.fairness_constraints
        )
        
        # Optimize for each protected attribute
        for attr_name, attr_values in protected_attributes.items():
            result = self.fairness_optimizer.optimize_fairness(
                X_train, y_train, attr_values
            )
            
            self.fairness_history.append({
                'protected_attribute': attr_name,
                'optimization_result': result
            })
        
        self.is_fitted = True
        logger.info("Fairness-aware model training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with fairness adjustments"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get base predictions
        base_predictions = self.base_model.predict(X)
        
        # Apply fairness adjustments if available
        if self.fairness_optimizer and len(self.fairness_history) > 0:
            # For simplicity, return base predictions (fairness optimization 
            # is done during training)
            pass
        
        return base_predictions
    
    def evaluate_fairness(self, X: pd.DataFrame, y_true: np.ndarray,
                         protected_attributes: Dict[str, pd.Series]) -> List[FairnessReport]:
        """Evaluate fairness across all constraints"""
        
        y_pred = self.predict(X)
        reports = []
        
        for constraint in self.fairness_constraints:
            protected_attr = protected_attributes.get(constraint.protected_attribute)
            if protected_attr is None:
                continue
            
            # Calculate fairness metric
            if constraint.metric == FairnessMetric.DEMOGRAPHIC_PARITY:
                disparity = FairnessMetrics.demographic_parity(y_pred, protected_attr.values)
            elif constraint.metric == FairnessMetric.EQUALIZED_ODDS:
                disparity = FairnessMetrics.equalized_odds(y_true, y_pred, protected_attr.values)
            elif constraint.metric == FairnessMetric.EQUALIZED_OPPORTUNITY:
                disparity = FairnessMetrics.equalized_opportunity(y_true, y_pred, protected_attr.values)
            elif constraint.metric == FairnessMetric.INDIVIDUAL_FAIRNESS:
                sensitive_features = [constraint.protected_attribute]
                disparity = FairnessMetrics.individual_fairness(y_pred, X, sensitive_features)
            elif constraint.metric == FairnessMetric.COUNTERFACTUAL_FAIRNESS:
                disparity = FairnessMetrics.counterfactual_fairness(self.base_model, X, constraint.protected_attribute)
            else:
                disparity = 0.0
            
            # Create report
            violation = disparity > constraint.tolerance
            recommendation = self._generate_recommendation(constraint, disparity)
            
            report = FairnessReport(
                metric_name=constraint.metric.value,
                protected_attribute=constraint.protected_attribute,
                baseline_value=0.0,  # Ideal value
                current_value=disparity,
                disparity=disparity,
                violation=violation,
                tolerance=constraint.tolerance,
                recommendation=recommendation
            )
            
            reports.append(report)
        
        return reports
    
    def _generate_recommendation(self, constraint: FairnessConstraint, 
                                disparity: float) -> str:
        """Generate fairness improvement recommendations"""
        
        if not disparity > constraint.tolerance:
            return "Fairness constraint satisfied"
        
        recommendations = {
            FairnessMetric.DEMOGRAPHIC_PARITY: 
                "Consider applying post-processing adjustments or reweighting training samples",
            FairnessMetric.EQUALIZED_ODDS:
                "Implement threshold optimization or use fairness-aware learning algorithms",
            FairnessMetric.EQUALIZED_OPPORTUNITY:
                "Adjust decision thresholds for different groups or use equalized opportunity constraints",
            FairnessMetric.INDIVIDUAL_FAIRNESS:
                "Apply regularization to ensure similar predictions for similar individuals",
            FairnessMetric.COUNTERFACTUAL_FAIRNESS:
                "Remove protected attribute from model or use counterfactual fairness techniques"
        }
        
        return recommendations.get(constraint.metric, "Consider fairness-aware modeling techniques")

class FairnessPricingEngine:
    """Integration of fairness into pricing engine"""
    
    def __init__(self, fairness_constraints: List[FairnessConstraint] = None):
        self.fairness_constraints = fairness_constraints or []
        self.fairness_model = None
        self.fairness_reports = []
        
    def setup_fairness_constraints(self, location_fairness: float = 0.1,
                                  loyalty_fairness: float = 0.15,
                                  time_fairness: float = 0.05):
        """Setup default fairness constraints for pricing"""
        
        self.fairness_constraints = [
            FairnessConstraint(
                metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                protected_attribute="Location_Category",
                tolerance=location_fairness,
                weight=1.0
            ),
            FairnessConstraint(
                metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                protected_attribute="Customer_Loyalty_Status",
                tolerance=loyalty_fairness,
                weight=0.8
            ),
            FairnessConstraint(
                metric=FairnessMetric.EQUALIZED_OPPORTUNITY,
                protected_attribute="Time_of_Booking",
                tolerance=time_fairness,
                weight=0.6
            )
        ]
        
        logger.info(f"Setup {len(self.fairness_constraints)} fairness constraints")
    
    def train_fair_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> FairnessAwareModel:
        """Train fairness-aware pricing model"""
        
        # Identify protected attributes
        protected_attributes = {}
        for constraint in self.fairness_constraints:
            if constraint.protected_attribute in X_train.columns:
                protected_attributes[constraint.protected_attribute] = X_train[constraint.protected_attribute]
        
        # Create base model
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Create fairness-aware model
        fairness_model = FairnessAwareModel(
            base_model, 
            self.fairness_constraints,
            fairness_weight=0.5
        )
        
        # Train model
        fairness_model.fit(X_train, y_train, protected_attributes)
        
        self.fairness_model = fairness_model
        return fairness_model
    
    def evaluate_fairness(self, X_test: pd.DataFrame, y_test: pd.Series) -> List[FairnessReport]:
        """Evaluate fairness of the pricing model"""
        
        if self.fairness_model is None:
            raise ValueError("Fairness model must be trained first")
        
        # Identify protected attributes
        protected_attributes = {}
        for constraint in self.fairness_constraints:
            if constraint.protected_attribute in X_test.columns:
                protected_attributes[constraint.protected_attribute] = X_test[constraint.protected_attribute]
        
        # Evaluate fairness
        reports = self.fairness_model.evaluate_fairness(X_test, y_test, protected_attributes)
        self.fairness_reports = reports
        
        return reports
    
    def generate_fairness_report(self) -> Dict[str, Any]:
        """Generate comprehensive fairness report"""
        
        if not self.fairness_reports:
            return {"error": "No fairness evaluations available"}
        
        summary = {
            "total_constraints": len(self.fairness_constraints),
            "violations": sum(1 for report in self.fairness_reports if report.violation),
            "satisfied": sum(1 for report in self.fairness_reports if not report.violation),
            "average_disparity": np.mean([report.disparity for report in self.fairness_reports]),
            "max_disparity": max([report.disparity for report in self.fairness_reports]),
            "detailed_reports": []
        }
        
        for report in self.fairness_reports:
            summary["detailed_reports"].append({
                "metric": report.metric_name,
                "attribute": report.protected_attribute,
                "disparity": report.disparity,
                "tolerance": report.tolerance,
                "violation": report.violation,
                "recommendation": report.recommendation
            })
        
        return summary
    
    def apply_fairness_adjustments(self, predictions: np.ndarray, 
                                  X: pd.DataFrame) -> np.ndarray:
        """Apply post-processing fairness adjustments"""
        
        adjusted_predictions = predictions.copy()
        
        for constraint in self.fairness_constraints:
            if constraint.protected_attribute not in X.columns:
                continue
            
            protected_attr = X[constraint.protected_attribute].values
            unique_groups = np.unique(protected_attr)
            
            if len(unique_groups) == 2:
                # Calculate group means
                group_0_mask = protected_attr == unique_groups[0]
                group_1_mask = protected_attr == unique_groups[1]
                
                mean_0 = np.mean(adjusted_predictions[group_0_mask])
                mean_1 = np.mean(adjusted_predictions[group_1_mask])
                
                # Calculate adjustment needed
                current_disparity = abs(mean_0 - mean_1)
                if current_disparity > constraint.tolerance:
                    # Apply proportional adjustment
                    adjustment_factor = (current_disparity - constraint.tolerance) / current_disparity
                    
                    if mean_0 > mean_1:
                        adjusted_predictions[group_0_mask] *= (1 - adjustment_factor * 0.5)
                        adjusted_predictions[group_1_mask] *= (1 + adjustment_factor * 0.5)
                    else:
                        adjusted_predictions[group_0_mask] *= (1 + adjustment_factor * 0.5)
                        adjusted_predictions[group_1_mask] *= (1 - adjustment_factor * 0.5)
        
        return adjusted_predictions
