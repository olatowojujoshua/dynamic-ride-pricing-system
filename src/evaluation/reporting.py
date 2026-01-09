"""
Comprehensive reporting for Dynamic Ride Pricing System evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..config import REPORTS_DIR, FIGURES_DIR
from ..utils.logger import logger

class EvaluationReporter:
    """
    Generate comprehensive evaluation reports for the dynamic pricing system
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize evaluation reporter
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or REPORTS_DIR
        self.figures_dir = output_dir / "figures" if output_dir else FIGURES_DIR
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized evaluation reporter with output directory: {self.output_dir}")
    
    def generate_model_evaluation_report(self, 
                                       baseline_results: Dict[str, Any],
                                       surge_results: Dict[str, Any],
                                       comparison_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive model evaluation report
        
        Args:
            baseline_results: Baseline model evaluation results
            surge_results: Surge model evaluation results
            comparison_results: Model comparison results
        
        Returns:
            Path to generated report file
        """
        report_content = []
        report_content.append("# DYNAMIC PRICING MODEL EVALUATION REPORT")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Executive Summary
        report_content.append("## EXECUTIVE SUMMARY")
        if comparison_results.get('best_model'):
            best_model = comparison_results['best_model']
            report_content.append(f"- **Best Performing Model**: {best_model}")
        
        if 'baseline_metrics' in baseline_results:
            baseline_mape = baseline_results['baseline_metrics'].get('mape', 0)
            report_content.append(f"- **Baseline Model MAPE**: {baseline_mape:.3f}")
        
        if 'surge_metrics' in surge_results:
            surge_pinball = surge_results['surge_metrics'].get('pinball_loss', 0)
            report_content.append(f"- **Surge Model Pinball Loss**: {surge_pinball:.3f}")
        
        report_content.append("")
        
        # Baseline Model Performance
        report_content.append("## BASELINE MODEL PERFORMANCE")
        if 'baseline_metrics' in baseline_results:
            metrics = baseline_results['baseline_metrics']
            report_content.append("### Key Metrics:")
            report_content.append(f"- **RMSE**: {metrics.get('rmse', 0):.3f}")
            report_content.append(f"- **MAE**: {metrics.get('mae', 0):.3f}")
            report_content.append(f"- **MAPE**: {metrics.get('mape', 0):.3f}")
            report_content.append(f"- **R²**: {metrics.get('r2', 0):.3f}")
            report_content.append(f"- **Accuracy within 10%**: {metrics.get('accuracy_within_10_percent', 0):.1f}%")
            
            if 'feature_importance' in baseline_results:
                report_content.append("\n### Top Features:")
                for feature, importance in list(baseline_results['feature_importance'].items())[:10]:
                    report_content.append(f"- **{feature}**: {importance:.4f}")
        
        report_content.append("")
        
        # Surge Model Performance
        report_content.append("## SURGE MODEL PERFORMANCE")
        if 'surge_metrics' in surge_results:
            metrics = surge_results['surge_metrics']
            report_content.append("### Key Metrics:")
            report_content.append(f"- **Pinball Loss**: {metrics.get('pinball_loss', 0):.3f}")
            report_content.append(f"- **Quantile Coverage**: {metrics.get('quantile_coverage', 0):.1f}%")
            report_content.append(f"- **Coverage Error**: {metrics.get('coverage_error', 0):.1f}%")
            
            if 'feature_importance' in surge_results:
                report_content.append("\n### Top Features:")
                for feature, importance in list(surge_results['feature_importance'].items())[:10]:
                    report_content.append(f"- **{feature}**: {importance:.4f}")
        
        report_content.append("")
        
        # Model Comparison
        report_content.append("## MODEL COMPARISON")
        if 'model_rankings' in comparison_results:
            for metric, ranking in comparison_results['model_rankings'].items():
                report_content.append(f"### {metric.upper()} Ranking:")
                for i, model in enumerate(ranking[:5], 1):
                    report_content.append(f"{i}. {model}")
                report_content.append("")
        
        # Recommendations
        report_content.append("## RECOMMENDATIONS")
        recommendations = self._generate_model_recommendations(baseline_results, surge_results, comparison_results)
        for rec in recommendations:
            report_content.append(f"- {rec}")
        
        # Save report
        report_path = self.output_dir / "model_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Model evaluation report saved to {report_path}")
        return str(report_path)
    
    def generate_business_impact_report(self, 
                                      revenue_results: Dict[str, Any],
                                      stability_results: Dict[str, Any],
                                      fairness_results: Dict[str, Any]) -> str:
        """
        Generate business impact report
        
        Args:
            revenue_results: Revenue simulation results
            stability_results: Price stability analysis
            fairness_results: Fairness analysis results
        
        Returns:
            Path to generated report file
        """
        report_content = []
        report_content.append("# DYNAMIC PRICING BUSINESS IMPACT REPORT")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Executive Summary
        report_content.append("## EXECUTIVE SUMMARY")
        
        if 'revenue_lift_percentage' in revenue_results:
            revenue_lift = revenue_results['revenue_lift_percentage']
            report_content.append(f"- **Revenue Lift**: {revenue_lift:.1f}%")
            
            if revenue_lift > 10:
                report_content.append("- **Assessment**: Strong positive impact")
            elif revenue_lift > 5:
                report_content.append("- **Assessment**: Moderate positive impact")
            else:
                report_content.append("- **Assessment**: Limited impact - review strategy")
        
        if 'stability_score' in stability_results:
            stability_score = stability_results['stability_score']
            report_content.append(f"- **Price Stability**: {stability_score:.1f}/100")
            
            if stability_score > 80:
                report_content.append("- **Assessment**: Excellent stability")
            elif stability_score > 60:
                report_content.append("- **Assessment**: Good stability")
            else:
                report_content.append("- **Assessment**: Stability concerns - review algorithms")
        
        if 'overall_fairness_score' in fairness_results:
            fairness_score = fairness_results['overall_fairness_score']
            report_content.append(f"- **Fairness Score**: {fairness_score:.1f}/100")
            
            if fairness_score > 80:
                report_content.append("- **Assessment**: Excellent fairness")
            elif fairness_score > 60:
                report_content.append("- **Assessment**: Good fairness")
            else:
                report_content.append("- **Assessment**: Fairness concerns - review policies")
        
        report_content.append("")
        
        # Revenue Analysis
        report_content.append("## REVENUE ANALYSIS")
        if 'baseline_revenue' in revenue_results:
            report_content.append("### Revenue Comparison:")
            report_content.append(f"- **Baseline Revenue**: ${revenue_results['baseline_revenue']:,.2f}")
            report_content.append(f"- **Dynamic Revenue**: ${revenue_results['dynamic_revenue']:,.2f}")
            report_content.append(f"- **Revenue Difference**: ${revenue_results['revenue_difference']:,.2f}")
            report_content.append(f"- **Revenue Lift**: {revenue_results['revenue_lift_percentage']:.1f}%")
            report_content.append(f"- **Price Change**: {revenue_results['price_change_percentage']:.1f}%")
            report_content.append(f"- **Demand Change**: {revenue_results['demand_change_percentage']:.1f}%")
        
        if 'revenue_by_location' in revenue_results:
            report_content.append("\n### Revenue by Location:")
            for location, metrics in revenue_results['revenue_by_location'].items():
                report_content.append(f"- **{location}**: {metrics['revenue_lift_percentage']:.1f}% lift")
        
        report_content.append("")
        
        # Stability Analysis
        report_content.append("## STABILITY ANALYSIS")
        if 'overall_volatility' in stability_results:
            volatility = stability_results['overall_volatility']
            report_content.append("### Volatility Metrics:")
            report_content.append(f"- **Price Standard Deviation**: ${volatility['price_std']:.2f}")
            report_content.append(f"- **Coefficient of Variation**: {volatility['coefficient_of_variation']:.3f}")
            report_content.append(f"- **Price Range**: ${volatility['price_range']:.2f}")
            report_content.append(f"- **Stability Score**: {stability_results.get('stability_score', 0):.1f}/100")
        
        if 'time_period_analysis' in stability_results:
            report_content.append("\n### Stability by Time Period:")
            for period, analysis in stability_results['time_period_analysis'].items():
                cv = analysis['volatility']['coefficient_of_variation']
                report_content.append(f"- **{period}**: CV = {cv:.3f}")
        
        report_content.append("")
        
        # Fairness Analysis
        report_content.append("## FAIRNESS ANALYSIS")
        if 'fairness_scores' in fairness_results:
            scores = fairness_results['fairness_scores']
            report_content.append("### Fairness Scores:")
            for metric, score in scores.items():
                report_content.append(f"- **{metric.replace('_', ' ').title()}**: {score:.3f}")
        
        if 'violations' in fairness_results:
            violations = fairness_results['violations']
            report_content.append("\n### Fairness Violations:")
            for category, violation_list in violations.items():
                if violation_list:
                    report_content.append(f"- **{category.replace('_', ' ').title()}**:")
                    for violation in violation_list:
                        report_content.append(f"  - {violation}")
        
        report_content.append("")
        
        # Recommendations
        report_content.append("## BUSINESS RECOMMENDATIONS")
        recommendations = self._generate_business_recommendations(revenue_results, stability_results, fairness_results)
        for rec in recommendations:
            report_content.append(f"- {rec}")
        
        # Save report
        report_path = self.output_dir / "business_impact_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Business impact report saved to {report_path}")
        return str(report_path)
    
    def generate_comprehensive_report(self, 
                                     model_results: Dict[str, Any],
                                     business_results: Dict[str, Any],
                                     system_config: Dict[str, Any]) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            model_results: Model evaluation results
            business_results: Business impact results
            system_config: System configuration
        
        Returns:
            Path to generated report file
        """
        report_content = []
        report_content.append("# COMPREHENSIVE DYNAMIC PRICING EVALUATION REPORT")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Table of Contents
        report_content.append("## TABLE OF CONTENTS")
        report_content.append("1. [Executive Summary](#executive-summary)")
        report_content.append("2. [Model Performance](#model-performance)")
        report_content.append("3. [Business Impact](#business-impact)")
        report_content.append("4. [System Configuration](#system-configuration)")
        report_content.append("5. [Implementation Roadmap](#implementation-roadmap)")
        report_content.append("6. [Risk Assessment](#risk-assessment)")
        report_content.append("")
        
        # Executive Summary
        report_content.append("## EXECUTIVE SUMMARY")
        executive_summary = self._generate_executive_summary(model_results, business_results)
        for point in executive_summary:
            report_content.append(f"- {point}")
        report_content.append("")
        
        # Model Performance
        report_content.append("## MODEL PERFORMANCE")
        if 'baseline_results' in model_results:
            baseline = model_results['baseline_results']
            if 'baseline_metrics' in baseline:
                metrics = baseline['baseline_metrics']
                report_content.append("### Baseline Model:")
                report_content.append(f"- **MAPE**: {metrics.get('mape', 0):.3f}")
                report_content.append(f"- **R²**: {metrics.get('r2', 0):.3f}")
                report_content.append(f"- **Performance Grade**: {baseline.get('performance_grade', 'N/A')}")
        
        if 'surge_results' in model_results:
            surge = model_results['surge_results']
            if 'surge_metrics' in surge:
                metrics = surge['surge_metrics']
                report_content.append("### Surge Model:")
                report_content.append(f"- **Pinball Loss**: {metrics.get('pinball_loss', 0):.3f}")
                report_content.append(f"- **Coverage**: {metrics.get('quantile_coverage', 0):.1f}%")
                report_content.append(f"- **Performance Grade**: {surge.get('performance_grade', 'N/A')}")
        
        report_content.append("")
        
        # Business Impact
        report_content.append("## BUSINESS IMPACT")
        if 'revenue_results' in business_results:
            revenue = business_results['revenue_results']
            report_content.append("### Revenue Impact:")
            report_content.append(f"- **Revenue Lift**: {revenue.get('revenue_lift_percentage', 0):.1f}%")
            report_content.append(f"- **Demand Impact**: {revenue.get('demand_change_percentage', 0):.1f}%")
        
        if 'stability_results' in business_results:
            stability = business_results['stability_results']
            report_content.append("### Price Stability:")
            report_content.append(f"- **Stability Score**: {stability.get('stability_score', 0):.1f}/100")
        
        if 'fairness_results' in business_results:
            fairness = business_results['fairness_results']
            report_content.append("### Fairness:")
            report_content.append(f"- **Overall Score**: {fairness.get('overall_fairness_score', 0):.1f}/100")
        
        report_content.append("")
        
        # System Configuration
        report_content.append("## SYSTEM CONFIGURATION")
        if 'model_config' in system_config:
            config = system_config['model_config']
            report_content.append("### Model Configuration:")
            for key, value in config.items():
                report_content.append(f"- **{key}**: {value}")
        
        if 'constraint_config' in system_config:
            config = system_config['constraint_config']
            report_content.append("### Pricing Constraints:")
            for key, value in config.items():
                report_content.append(f"- **{key}**: {value}")
        
        report_content.append("")
        
        # Implementation Roadmap
        report_content.append("## IMPLEMENTATION ROADMAP")
        roadmap = self._generate_implementation_roadmap(model_results, business_results)
        for phase, items in roadmap.items():
            report_content.append(f"### {phase}")
            for item in items:
                report_content.append(f"- {item}")
            report_content.append("")
        
        # Risk Assessment
        report_content.append("## RISK ASSESSMENT")
        risks = self._generate_risk_assessment(model_results, business_results)
        for risk in risks:
            report_content.append(f"- {risk}")
        
        # Save comprehensive report
        report_path = self.output_dir / "comprehensive_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Comprehensive evaluation report saved to {report_path}")
        return str(report_path)
    
    def save_evaluation_data(self, data: Dict[str, Any], filename: str) -> str:
        """
        Save evaluation data as JSON
        
        Args:
            data: Data to save
            filename: Filename for the saved data
        
        Returns:
            Path to saved file
        """
        data_path = self.output_dir / f"{filename}.json"
        
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Evaluation data saved to {data_path}")
        return str(data_path)
    
    def _generate_model_recommendations(self, baseline_results: Dict[str, Any], 
                                      surge_results: Dict[str, Any],
                                      comparison_results: Dict[str, Any]) -> List[str]:
        """Generate model-specific recommendations"""
        recommendations = []
        
        # Baseline model recommendations
        if 'baseline_metrics' in baseline_results:
            mape = baseline_results['baseline_metrics'].get('mape', 0)
            r2 = baseline_results['baseline_metrics'].get('r2', 0)
            
            if mape > 0.15:
                recommendations.append("Consider feature engineering to improve baseline model accuracy")
            if r2 < 0.8:
                recommendations.append("Explore ensemble methods or more complex models")
        
        # Surge model recommendations
        if 'surge_metrics' in surge_results:
            coverage_error = surge_results['surge_metrics'].get('coverage_error', 0)
            if coverage_error > 10:
                recommendations.append("Adjust quantile regression parameters for better coverage")
        
        # Model comparison recommendations
        if comparison_results.get('best_model'):
            recommendations.append(f"Deploy {comparison_results['best_model']} as the primary model")
        
        return recommendations
    
    def _generate_business_recommendations(self, revenue_results: Dict[str, Any],
                                         stability_results: Dict[str, Any],
                                         fairness_results: Dict[str, Any]) -> List[str]:
        """Generate business-specific recommendations"""
        recommendations = []
        
        # Revenue recommendations
        if revenue_results.get('revenue_lift_percentage', 0) < 5:
            recommendations.append("Review pricing strategy - revenue lift is minimal")
        elif revenue_results.get('revenue_lift_percentage', 0) > 20:
            recommendations.append("Monitor customer satisfaction - high revenue lift may impact demand")
        
        # Stability recommendations
        if stability_results.get('stability_score', 100) < 60:
            recommendations.append("Implement price smoothing mechanisms to improve stability")
        
        # Fairness recommendations
        if fairness_results.get('overall_fairness_score', 100) < 70:
            recommendations.append("Review fairness policies and adjust pricing constraints")
        
        return recommendations
    
    def _generate_executive_summary(self, model_results: Dict[str, Any],
                                   business_results: Dict[str, Any]) -> List[str]:
        """Generate executive summary points"""
        summary = []
        
        # Model performance summary
        if 'baseline_results' in model_results:
            grade = model_results['baseline_results'].get('performance_grade', 'N/A')
            summary.append(f"Baseline model achieved {grade} grade performance")
        
        # Business impact summary
        if 'revenue_results' in business_results:
            lift = business_results['revenue_results'].get('revenue_lift_percentage', 0)
            summary.append(f"Dynamic pricing system projected to deliver {lift:.1f}% revenue lift")
        
        # Overall assessment
        if 'stability_results' in business_results and 'fairness_results' in business_results:
            stability = business_results['stability_results'].get('stability_score', 0)
            fairness = business_results['fairness_results'].get('overall_fairness_score', 0)
            
            if stability > 70 and fairness > 70:
                summary.append("System demonstrates strong stability and fairness characteristics")
            else:
                summary.append("System requires optimization for stability and/or fairness")
        
        return summary
    
    def _generate_implementation_roadmap(self, model_results: Dict[str, Any],
                                        business_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate implementation roadmap"""
        roadmap = {
            "Phase 1 - Foundation": [
                "Deploy baseline model for initial pricing",
                "Implement basic constraint system",
                "Set up monitoring and alerting"
            ],
            "Phase 2 - Enhancement": [
                "Integrate surge pricing model",
                "Implement fairness adjustments",
                "Add real-time monitoring dashboard"
            ],
            "Phase 3 - Optimization": [
                "Enable dynamic learning and model updates",
                "Implement advanced constraint management",
                "Add A/B testing capabilities"
            ],
            "Phase 4 - Scale": [
                "Scale to all markets and vehicle types",
                "Implement predictive surge pricing",
                "Add customer personalization features"
            ]
        }
        
        return roadmap
    
    def _generate_risk_assessment(self, model_results: Dict[str, Any],
                                   business_results: Dict[str, Any]) -> List[str]:
        """Generate risk assessment"""
        risks = []
        
        # Model risks
        if 'baseline_results' in model_results:
            mape = model_results['baseline_results'].get('baseline_metrics', {}).get('mape', 0)
            if mape > 0.2:
                risks.append("High prediction error may lead to customer dissatisfaction")
        
        # Business risks
        if 'revenue_results' in business_results:
            demand_change = business_results['revenue_results'].get('demand_change_percentage', 0)
            if demand_change < -10:
                risks.append("Significant demand reduction may impact market share")
        
        # Stability risks
        if 'stability_results' in business_results:
            stability_score = business_results['stability_results'].get('stability_score', 100)
            if stability_score < 50:
                risks.append("Price volatility may lead to customer confusion")
        
        # Fairness risks
        if 'fairness_results' in business_results:
            violations_count = sum(len(v) for v in business_results['fairness_results'].get('violations', {}).values())
            if violations_count > 5:
                risks.append("Multiple fairness violations may lead to regulatory issues")
        
        return risks
