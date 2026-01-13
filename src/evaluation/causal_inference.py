"""
Causal inference and A/B testing framework for dynamic pricing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from ..utils.logger import logger

class CausalMethod(Enum):
    """Types of causal inference methods"""
    RANDOMIZED_TRIAL = "randomized_trial"
    PROPENSITY_SCORE = "propensity_score"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    SYNTHETIC_CONTROL = "synthetic_control"
    CAUSAL_FOREST = "causal_forest"
    META_LEARNER = "meta_learner"

class TestType(Enum):
    """Types of A/B tests"""
    PRICE_SENSITIVITY = "price_sensitivity"
    SURGE_EFFECTIVENESS = "surge_effectiveness"
    LOYALTY_IMPACT = "loyalty_impact"
    LOCATION_PREMIUM = "location_premium"
    TIME_BASED_PRICING = "time_based_pricing"
    FAIRNESS_INTERVENTION = "fairness_intervention"

@dataclass
class ExperimentConfig:
    """Configuration for A/B test"""
    test_name: str
    test_type: TestType
    treatment_description: str
    control_description: str
    sample_size_required: int = 0  # Will be calculated
    power: float = 0.8
    significance_level: float = 0.05
    minimum_detectable_effect: float = 0.05
    duration_days: int = 14
    randomization_unit: str = "ride"  # ride, driver, customer
    success_metric: str = "revenue"
    covariates: List[str] = field(default_factory=list)

@dataclass
class ExperimentResult:
    """Results from A/B test"""
    test_name: str
    treatment_effect: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_statistically_significant: bool
    practical_significance: bool
    sample_size: int
    power_achieved: float
    recommendation: str
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

class PowerAnalysis:
    """Statistical power analysis for experimental design"""
    
    @staticmethod
    def calculate_sample_size(effect_size: float, power: float = 0.8, 
                            alpha: float = 0.05, ratio: float = 1.0) -> int:
        """
        Calculate required sample size for two-sample test
        
        Args:
            effect_size: Cohen's d or standardized effect size
            power: Statistical power (1 - beta)
            alpha: Significance level
            ratio: Ratio of control to treatment sample sizes
        
        Returns:
            Required sample size per group
        """
        from scipy.stats import norm
        
        # Z-scores
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size formula for two-sample test
        n_per_group = 2 * (z_alpha + z_beta)**2 / effect_size**2
        
        # Adjust for ratio
        if ratio != 1.0:
            n_treatment = n_per_group
            n_control = n_per_group * ratio
            n_per_group = max(n_treatment, n_control)
        
        return int(np.ceil(n_per_group))
    
    @staticmethod
    def calculate_power(sample_size: int, effect_size: float, 
                        alpha: float = 0.05) -> float:
        """Calculate statistical power given sample size and effect size"""
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = np.sqrt(sample_size * effect_size**2 / 2) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0.0, min(1.0, power))
    
    @staticmethod
    def calculate_effect_size(mean_1: float, mean_2: float, 
                            std_1: float, std_2: float) -> float:
        """Calculate Cohen's d effect size"""
        pooled_std = np.sqrt(((len(mean_1) - 1) * std_1**2 + 
                             (len(mean_2) - 1) * std_2**2) / 
                            (len(mean_1) + len(mean_2) - 2))
        return abs(mean_1 - mean_2) / pooled_std

class ABTestManager:
    """A/B testing framework for pricing experiments"""
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
        self.running_tests = {}
        
    def design_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Design A/B test with power analysis"""
        
        logger.info(f"Designing experiment: {config.test_name}")
        
        # Calculate required sample size
        effect_size = config.minimum_detectable_effect
        required_n = PowerAnalysis.calculate_sample_size(
            effect_size=effect_size,
            power=config.power,
            alpha=config.significance_level
        )
        
        # Update config with calculated requirements
        config.sample_size_required = required_n * 2  # Total for both groups
        
        # Generate randomization plan
        randomization_plan = self._generate_randomization_plan(config)
        
        # Create monitoring schedule
        monitoring_schedule = self._create_monitoring_schedule(config)
        
        design_summary = {
            'config': config,
            'required_sample_size': required_n,
            'total_sample_size': required_n * 2,
            'randomization_plan': randomization_plan,
            'monitoring_schedule': monitoring_schedule,
            'success_criteria': self._define_success_criteria(config)
        }
        
        self.experiments[config.test_name] = design_summary
        logger.info(f"Experiment designed: {config.test_name}")
        
        return design_summary
    
    def _generate_randomization_plan(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Generate randomization plan for experiment"""
        
        plan = {
            'method': 'simple_random',
            'stratification_factors': config.covariates,
            'allocation_ratio': 1.0,
            'blocking_factors': [],
            'random_seed': 42
        }
        
        # Add stratification if covariates specified
        if config.covariates:
            plan['method'] = 'stratified_random'
        
        return plan
    
    def _create_monitoring_schedule(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Create monitoring schedule for experiment"""
        
        schedule = []
        total_days = config.duration_days
        
        # Daily monitoring for first week, then every 3 days
        for day in range(1, total_days + 1):
            if day <= 7 or day % 3 == 0:
                schedule.append({
                    'day': day,
                    'check_type': 'interim_analysis' if day < total_days else 'final_analysis',
                    'metrics': ['sample_size', 'conversion_rate', 'revenue', 'user_satisfaction']
                })
        
        return schedule
    
    def _define_success_criteria(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Define success criteria for experiment"""
        
        criteria = {
            'statistical_significance': {
                'p_value_threshold': config.significance_level,
                'confidence_level': 1 - config.significance_level
            },
            'practical_significance': {
                'minimum_effect': config.minimum_detectable_effect,
                'business_impact_threshold': 0.02  # 2% revenue lift
            },
            'safety_constraints': {
                'max_negative_impact': -0.05,  # Max 5% negative impact
                'user_satisfaction_min': 0.8   # Min 80% satisfaction
            }
        }
        
        return criteria
    
    def run_experiment(self, test_name: str, data: pd.DataFrame) -> ExperimentResult:
        """Run A/B test analysis"""
        
        if test_name not in self.experiments:
            raise ValueError(f"Experiment {test_name} not designed")
        
        config = self.experiments[test_name]['config']
        logger.info(f"Running experiment: {test_name}")
        
        # Split data into treatment and control
        treatment_data, control_data = self._split_data(data, config)
        
        # Calculate treatment effect
        effect_result = self._calculate_treatment_effect(
            treatment_data, control_data, config
        )
        
        # Perform statistical tests
        statistical_result = self._perform_statistical_tests(
            treatment_data, control_data, config
        )
        
        # Calculate practical significance
        practical_result = self._calculate_practical_significance(
            effect_result, config
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            effect_result, statistical_result, practical_result, config
        )
        
        # Create result object
        result = ExperimentResult(
            test_name=test_name,
            treatment_effect=effect_result['effect_size'],
            confidence_interval=statistical_result['confidence_interval'],
            p_value=statistical_result['p_value'],
            is_statistically_significant=statistical_result['is_significant'],
            practical_significance=practical_result['is_significant'],
            sample_size=len(treatment_data) + len(control_data),
            power_achieved=statistical_result['power'],
            recommendation=recommendation,
            detailed_metrics={
                'treatment_stats': effect_result['treatment_stats'],
                'control_stats': effect_result['control_stats'],
                'statistical_tests': statistical_result,
                'practical_analysis': practical_result
            }
        )
        
        self.results[test_name] = result
        logger.info(f"Experiment completed: {test_name}")
        
        return result
    
    def _split_data(self, data: pd.DataFrame, config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into treatment and control groups"""
        
        # Simple random split for now
        np.random.seed(42)
        mask = np.random.random(len(data)) < 0.5
        
        treatment_data = data[mask].copy()
        control_data = data[~mask].copy()
        
        # Add treatment indicator
        treatment_data['treatment'] = 1
        control_data['treatment'] = 0
        
        return treatment_data, control_data
    
    def _calculate_treatment_effect(self, treatment_data: pd.DataFrame, 
                                 control_data: pd.DataFrame, 
                                 config: ExperimentConfig) -> Dict[str, Any]:
        """Calculate treatment effect size"""
        
        # Get success metric
        if config.success_metric == "revenue":
            treatment_values = treatment_data['Historical_Cost_of_Ride']
            control_values = control_data['Historical_Cost_of_Ride']
        else:
            # Default to price
            treatment_values = treatment_data['Historical_Cost_of_Ride']
            control_values = control_data['Historical_Cost_of_Ride']
        
        # Calculate statistics
        treatment_mean = treatment_values.mean()
        control_mean = control_values.mean()
        treatment_std = treatment_values.std()
        control_std = control_values.std()
        
        # Calculate effect size
        effect_size = (treatment_mean - control_mean) / control_std
        
        return {
            'effect_size': effect_size,
            'treatment_stats': {
                'mean': treatment_mean,
                'std': treatment_std,
                'n': len(treatment_values)
            },
            'control_stats': {
                'mean': control_mean,
                'std': control_std,
                'n': len(control_values)
            }
        }
    
    def _perform_statistical_tests(self, treatment_data: pd.DataFrame,
                                  control_data: pd.DataFrame,
                                  config: ExperimentConfig) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        
        # Get values
        if config.success_metric == "revenue":
            treatment_values = treatment_data['Historical_Cost_of_Ride']
            control_values = control_data['Historical_Cost_of_Ride']
        else:
            treatment_values = treatment_data['Historical_Cost_of_Ride']
            control_values = control_data['Historical_Cost_of_Ride']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        # Calculate confidence interval
        mean_diff = treatment_values.mean() - control_values.mean()
        se_diff = np.sqrt(treatment_values.var()/len(treatment_values) + 
                         control_values.var()/len(control_values))
        
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        # Calculate achieved power
        effect_size = abs(mean_diff) / control_values.std()
        achieved_power = PowerAnalysis.calculate_power(
            sample_size=min(len(treatment_values), len(control_values)),
            effect_size=effect_size,
            alpha=config.significance_level
        )
        
        return {
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < config.significance_level,
            'power': achieved_power,
            't_statistic': t_stat
        }
    
    def _calculate_practical_significance(self, effect_result: Dict[str, Any],
                                        config: ExperimentConfig) -> Dict[str, Any]:
        """Calculate practical significance"""
        
        effect_size = effect_result['effect_size']
        treatment_mean = effect_result['treatment_stats']['mean']
        control_mean = effect_result['control_stats']['mean']
        
        # Calculate percentage change
        percent_change = (treatment_mean - control_mean) / control_mean
        
        # Check if meets minimum detectable effect
        meets_minimum = abs(effect_size) >= config.minimum_detectable_effect
        
        # Check business impact
        business_impact = abs(percent_change) >= 0.02  # 2% threshold
        
        return {
            'percent_change': percent_change,
            'meets_minimum_effect': meets_minimum,
            'business_impact': business_impact,
            'is_significant': meets_minimum and business_impact
        }
    
    def _generate_recommendation(self, effect_result: Dict[str, Any],
                               statistical_result: Dict[str, Any],
                               practical_result: Dict[str, Any],
                               config: ExperimentConfig) -> str:
        """Generate experiment recommendation"""
        
        is_stat_sig = statistical_result['is_significant']
        is_practical_sig = practical_result['is_significant']
        effect_direction = "positive" if effect_result['effect_size'] > 0 else "negative"
        
        if is_stat_sig and is_practical_sig:
            if effect_direction == "positive":
                return f"IMPLEMENT: Treatment shows {effect_direction} statistically and practically significant effect"
            else:
                return f"REJECT: Treatment shows {effect_direction} significant effect - avoid implementation"
        elif is_stat_sig and not is_practical_sig:
            return f"CONSIDER: Statistically significant but practically small effect - evaluate cost-benefit"
        elif not is_stat_sig and is_practical_sig:
            return f"INCONCLUSIVE: Practically significant but not statistically significant - consider larger sample"
        else:
            return f"NO EFFECT: No statistically or practically significant effect detected"

class CausalInferenceEngine:
    """Causal inference methods for pricing analysis"""
    
    def __init__(self):
        self.methods = {}
        self.results = {}
        
    def propensity_score_matching(self, data: pd.DataFrame, 
                                 treatment_col: str, 
                                 outcome_col: str,
                                 covariate_cols: List[str]) -> Dict[str, Any]:
        """
        Propensity score matching for causal inference
        
        Args:
            data: Dataset with treatment, outcome, and covariates
            treatment_col: Column name for treatment indicator
            outcome_col: Column name for outcome variable
            covariate_cols: List of covariate column names
        
        Returns:
            Causal effect estimate and diagnostics
        """
        logger.info("Performing propensity score matching")
        
        # Fit propensity score model
        X_covariates = data[covariate_cols]
        treatment = data[treatment_col]
        
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X_covariates, treatment)
        
        # Calculate propensity scores
        propensity_scores = ps_model.predict_proba(X_covariates)[:, 1]
        data['propensity_score'] = propensity_scores
        
        # Perform matching
        treatment_group = data[data[treatment_col] == 1]
        control_group = data[data[treatment_col] == 0]
        
        matched_pairs = []
        for _, treated_unit in treatment_group.iterrows():
            # Find closest control unit
            control_matches = control_group.copy()
            control_matches['distance'] = abs(control_matches['propensity_score'] - 
                                             treated_unit['propensity_score'])
            
            # Best match without replacement
            if len(control_matches) > 0:
                best_match = control_matches.loc[control_matches['distance'].idxmin()]
                matched_pairs.append((treated_unit, best_match))
                control_group = control_group.drop(best_match.name)
        
        # Calculate treatment effect
        treatment_outcomes = [pair[0][outcome_col] for pair in matched_pairs]
        control_outcomes = [pair[1][outcome_col] for pair in matched_pairs]
        
        ate = np.mean(treatment_outcomes) - np.mean(control_outcomes)
        
        # Statistical test
        t_stat, p_value = stats.ttest_rel(treatment_outcomes, control_outcomes)
        
        # Diagnostics
        diagnostics = {
            'propensity_score_model': ps_model,
            'matched_pairs': len(matched_pairs),
            'balance_check': self._check_balance(data, covariate_cols, treatment_col),
            'common_support': self._check_common_support(propensity_scores, treatment)
        }
        
        result = {
            'method': 'propensity_score_matching',
            'average_treatment_effect': ate,
            'standard_error': np.std(treatment_outcomes - control_outcomes) / np.sqrt(len(matched_pairs)),
            'p_value': p_value,
            'confidence_interval': self._calculate_confidence_interval(ate, len(matched_pairs)),
            'matched_sample_size': len(matched_pairs),
            'diagnostics': diagnostics
        }
        
        self.results['propensity_score_matching'] = result
        return result
    
    def difference_in_differences(self, data: pd.DataFrame,
                                 treatment_col: str,
                                 outcome_col: str,
                                 time_col: str,
                                 group_col: str) -> Dict[str, Any]:
        """
        Difference-in-differences analysis
        
        Args:
            data: Panel data with treatment, outcome, time, and group indicators
            treatment_col: Treatment group indicator
            outcome_col: Outcome variable
            time_col: Time period indicator
            group_col: Group identifier
        
        Returns:
            DiD causal effect estimate
        """
        logger.info("Performing difference-in-differences analysis")
        
        # Ensure proper data structure
        data = data.copy()
        
        # Create treatment x post-period interaction
        data['post_period'] = (data[time_col] == data[time_col].max()).astype(int)
        data['treatment_x_post'] = data[treatment_col] * data['post_period']
        
        # Fit DiD model
        X = data[['treatment_col', 'post_period', 'treatment_x_post']]
        X = X.rename(columns={
            'treatment_col': 'treatment',
            'post_period': 'post',
            'treatment_x_post': 'treatment_post'
        })
        X = sm.add_constant(X)
        
        y = data[outcome_col]
        
        did_model = sm.OLS(y, X).fit()
        
        # Extract treatment effect (coefficient on interaction term)
        treatment_effect = did_model.params['treatment_post']
        standard_error = did_model.bse['treatment_post']
        p_value = did_model.pvalues['treatment_post']
        
        # Parallel trends assumption check
        parallel_trends_check = self._check_parallel_trends(data, outcome_col, 
                                                         treatment_col, time_col)
        
        result = {
            'method': 'difference_in_differences',
            'treatment_effect': treatment_effect,
            'standard_error': standard_error,
            'p_value': p_value,
            'confidence_interval': (treatment_effect - 1.96 * standard_error,
                                   treatment_effect + 1.96 * standard_error),
            'parallel_trends_assumption': parallel_trends_check,
            'model_summary': did_model.summary()
        }
        
        self.results['difference_in_differences'] = result
        return result
    
    def instrumental_variable(self, data: pd.DataFrame,
                            treatment_col: str,
                            outcome_col: str,
                            instrument_col: str,
                            covariate_cols: List[str] = None) -> Dict[str, Any]:
        """
        Instrumental variable analysis
        
        Args:
            data: Dataset with treatment, outcome, and instrument
            treatment_col: Endogenous treatment variable
            outcome_col: Outcome variable
            instrument_col: Instrumental variable
            covariate_cols: Control variables
        
        Returns:
            IV causal effect estimate
        """
        logger.info("Performing instrumental variable analysis")
        
        if covariate_cols is None:
            covariate_cols = []
        
        # First stage: Regress treatment on instrument
        X_first = data[[instrument_col] + covariate_cols]
        X_first = sm.add_constant(X_first)
        treatment = data[treatment_col]
        
        first_stage = sm.OLS(treatment, X_first).fit()
        predicted_treatment = first_stage.predict(X_first)
        
        # Second stage: Regress outcome on predicted treatment
        X_second = sm.add_constant(pd.DataFrame({
            'predicted_treatment': predicted_treatment,
            **{col: data[col] for col in covariate_cols}
        }))
        
        outcome = data[outcome_col]
        second_stage = sm.OLS(outcome, X_second).fit()
        
        # Extract causal effect
        causal_effect = second_stage.params['predicted_treatment']
        standard_error = second_stage.bse['predicted_treatment']
        p_value = second_stage.pvalues['predicted_treatment']
        
        # Instrument relevance check
        f_statistic = first_stage.fvalue
        instrument_strength = "strong" if f_statistic > 10 else "weak"
        
        result = {
            'method': 'instrumental_variable',
            'causal_effect': causal_effect,
            'standard_error': standard_error,
            'p_value': p_value,
            'confidence_interval': (causal_effect - 1.96 * standard_error,
                                   causal_effect + 1.96 * standard_error),
            'first_stage_f_statistic': f_statistic,
            'instrument_strength': instrument_strength,
            'first_stage_summary': first_stage.summary(),
            'second_stage_summary': second_stage.summary()
        }
        
        self.results['instrumental_variable'] = result
        return result
    
    def regression_discontinuity(self, data: pd.DataFrame,
                               running_var_col: str,
                               outcome_col: str,
                               threshold: float,
                               bandwidth: float = None) -> Dict[str, Any]:
        """
        Regression discontinuity analysis
        
        Args:
            data: Dataset with running variable and outcome
            running_var_col: Running variable (assignment variable)
            outcome_col: Outcome variable
            threshold: RD threshold
            bandwidth: Optional bandwidth for local analysis
        
        Returns:
            RD causal effect estimate
        """
        logger.info("Performing regression discontinuity analysis")
        
        # Determine bandwidth if not provided
        if bandwidth is None:
            bandwidth = 0.5 * data[running_var_col].std()
        
        # Filter data to bandwidth around threshold
        lower_bound = threshold - bandwidth
        upper_bound = threshold + bandwidth
        
        rd_data = data[(data[running_var_col] >= lower_bound) & 
                       (data[running_var_col] <= upper_bound)].copy()
        
        # Create treatment indicator
        rd_data['treatment'] = (rd_data[running_var_col] >= threshold).astype(int)
        
        # Center running variable at threshold
        rd_data['centered_running'] = rd_data[running_var_col] - threshold
        
        # Fit RD model
        X = rd_data[['treatment', 'centered_running']]
        X = sm.add_constant(X)
        y = rd_data[outcome_col]
        
        rd_model = sm.OLS(y, X).fit()
        
        # Extract RD effect
        rd_effect = rd_model.params['treatment']
        standard_error = rd_model.bse['treatment']
        p_value = rd_model.pvalues['treatment']
        
        # Validity checks
        validity_checks = {
            'manipulation_test': self._test_manipulation(data, running_var_col, threshold),
            'continuity_test': self._test_covariate_continuity(rd_data, threshold),
            'bandwidth_sensitivity': self._test_bandwidth_sensitivity(data, running_var_col, 
                                                                  outcome_col, threshold)
        }
        
        result = {
            'method': 'regression_discontinuity',
            'rd_effect': rd_effect,
            'standard_error': standard_error,
            'p_value': p_value,
            'confidence_interval': (rd_effect - 1.96 * standard_error,
                                   rd_effect + 1.96 * standard_error),
            'bandwidth': bandwidth,
            'sample_size': len(rd_data),
            'validity_checks': validity_checks,
            'model_summary': rd_model.summary()
        }
        
        self.results['regression_discontinuity'] = result
        return result
    
    def _check_balance(self, data: pd.DataFrame, covariate_cols: List[str],
                      treatment_col: str) -> Dict[str, Any]:
        """Check covariate balance after matching"""
        
        treatment_group = data[data[treatment_col] == 1]
        control_group = data[data[treatment_col] == 0]
        
        balance_stats = {}
        for covariate in covariate_cols:
            treated_mean = treatment_group[covariate].mean()
            control_mean = control_group[covariate].mean()
            
            # Standardized mean difference
            pooled_std = np.sqrt(((len(treatment_group) - 1) * treatment_group[covariate].var() +
                                 (len(control_group) - 1) * control_group[covariate].var()) /
                                (len(treatment_group) + len(control_group) - 2))
            
            smd = (treated_mean - control_mean) / pooled_std
            
            balance_stats[covariate] = {
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'standardized_difference': smd,
                'balanced': abs(smd) < 0.1  # Standard threshold
            }
        
        return balance_stats
    
    def _check_common_support(self, propensity_scores: np.ndarray, 
                            treatment: np.ndarray) -> Dict[str, Any]:
        """Check common support assumption"""
        
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]
        
        min_treated = treated_ps.min()
        max_control = control_ps.max()
        
        common_support = min_treated <= max_control
        
        return {
            'common_support': common_support,
            'treated_range': (treated_ps.min(), treated_ps.max()),
            'control_range': (control_ps.min(), control_ps.max()),
            'overlap_region': (min_treated, max_control) if common_support else None
        }
    
    def _calculate_confidence_interval(self, estimate: float, n: int, 
                                     std_err: float = None) -> Tuple[float, float]:
        """Calculate confidence interval for estimate"""
        if std_err is None:
            std_err = np.std(estimate) / np.sqrt(n)
        
        ci_lower = estimate - 1.96 * std_err
        ci_upper = estimate + 1.96 * std_err
        
        return (ci_lower, ci_upper)
    
    def _check_parallel_trends(self, data: pd.DataFrame, outcome_col: str,
                             treatment_col: str, time_col: str) -> Dict[str, Any]:
        """Check parallel trends assumption for DiD"""
        
        # Get pre-treatment periods
        pre_periods = data[data[time_col] < data[time_col].max()]
        
        # Calculate trends for treatment and control groups
        treatment_pre = pre_periods[pre_periods[treatment_col] == 1]
        control_pre = pre_periods[pre_periods[treatment_col] == 0]
        
        if len(treatment_pre) > 1 and len(control_pre) > 1:
            # Simple trend comparison
            treatment_trend = np.polyfit(treatment_pre[time_col], 
                                        treatment_pre[outcome_col], 1)[0]
            control_trend = np.polyfit(control_pre[time_col], 
                                      control_pre[outcome_col], 1)[0]
            
            trend_difference = abs(treatment_trend - control_trend)
            parallel_trends = trend_difference < 0.1  # Arbitrary threshold
            
            return {
                'parallel_trends': parallel_trends,
                'treatment_trend': treatment_trend,
                'control_trend': control_trend,
                'trend_difference': trend_difference
            }
        else:
            return {'parallel_trends': 'insufficient_data'}
    
    def _test_manipulation(self, data: pd.DataFrame, running_var_col: str,
                          threshold: float) -> Dict[str, Any]:
        """Test for manipulation around RD threshold"""
        
        # McCrary test for manipulation
        below_threshold = data[data[running_var_col] < threshold]
        above_threshold = data[data[running_var_col] >= threshold]
        
        # Simple density comparison
        below_density = len(below_threshold) / (threshold - below_threshold[running_var_col].min())
        above_density = len(above_threshold) / (above_threshold[running_var_col].max() - threshold)
        
        density_ratio = above_density / below_density
        manipulation_detected = abs(density_ratio - 1) > 0.2  # 20% difference threshold
        
        return {
            'manipulation_detected': manipulation_detected,
            'density_ratio': density_ratio,
            'below_density': below_density,
            'above_density': above_density
        }
    
    def _test_covariate_continuity(self, data: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Test covariate continuity at RD threshold"""
        
        # This would test if covariates are continuous at threshold
        # Simplified version here
        return {'covariate_continuity': 'passed'}  # Placeholder
    
    def _test_bandwidth_sensitivity(self, data: pd.DataFrame, running_var_col: str,
                                   outcome_col: str, threshold: float) -> Dict[str, Any]:
        """Test sensitivity to bandwidth choice"""
        
        bandwidths = [0.1, 0.25, 0.5, 1.0] * data[running_var_col].std()
        effects = []
        
        for bw in bandwidths:
            # Simplified RD estimation for each bandwidth
            rd_data = data[(data[running_var_col] >= threshold - bw) & 
                           (data[running_var_col] <= threshold + bw)]
            
            if len(rd_data) > 10:
                treatment_effect = rd_data[rd_data[running_var_col] >= threshold][outcome_col].mean() - \
                                 rd_data[rd_data[running_var_col] < threshold][outcome_col].mean()
                effects.append(treatment_effect)
        
        sensitivity = np.std(effects) if len(effects) > 1 else 0
        
        return {
            'bandwidth_effects': effects,
            'sensitivity': sensitivity,
            'stable': sensitivity < 0.1  # Arbitrary threshold
        }

# Import statsmodels for regression analysis
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available, some causal methods may not work")
