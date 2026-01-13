"""
Model compression and computational complexity optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import time
import psutil
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import pickle
from pathlib import Path

from ..utils.logger import logger

class CompressionMethod(Enum):
    """Model compression methods"""
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    FEATURE_SELECTION = "feature_selection"
    LOW_RANK_APPROXIMATION = "low_rank_approximation"
    ENSEMBLE_COMPRESSION = "ensemble_compression"
    NEURAL_COMPRESSION = "neural_compression"

class OptimizationTarget(Enum):
    """Optimization targets"""
    LATENCY = "latency"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    SIZE = "size"
    THROUGHPUT = "throughput"

@dataclass
class CompressionConfig:
    """Configuration for model compression"""
    method: CompressionMethod
    target_metric: OptimizationTarget
    target_value: float
    max_accuracy_loss: float = 0.05  # Maximum 5% accuracy loss
    preserve_features: List[str] = field(default_factory=list)
    compression_ratio: float = 0.5  # Target compression ratio

@dataclass
class CompressionResult:
    """Results from model compression"""
    original_size: float
    compressed_size: float
    compression_ratio: float
    original_latency: float
    compressed_latency: float
    latency_improvement: float
    original_accuracy: float
    compressed_accuracy: float
    accuracy_loss: float
    memory_usage: Dict[str, float]
    compression_method: str
    success: bool
    recommendations: List[str] = field(default_factory=list)

class ModelProfiler:
    """Profile model performance and resource usage"""
    
    def __init__(self):
        self.profiles = {}
        
    def profile_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive model profiling"""
        
        logger.info(f"Profiling model: {model_name}")
        
        profile = {
            'model_name': model_name,
            'timestamp': time.time(),
            'size_metrics': self._measure_model_size(model),
            'latency_metrics': self._measure_latency(model, X_test),
            'memory_metrics': self._measure_memory_usage(model, X_test),
            'accuracy_metrics': self._measure_accuracy(model, X_test, y_test),
            'complexity_metrics': self._measure_complexity(model)
        }
        
        self.profiles[model_name] = profile
        return profile
    
    def _measure_model_size(self, model) -> Dict[str, float]:
        """Measure model size in different formats"""
        
        # Pickle size
        pickle_bytes = pickle.dumps(model)
        pickle_size = len(pickle_bytes) / (1024 * 1024)  # MB
        
        # Joblib size
        with open('temp_joblib.pkl', 'wb') as f:
            joblib.dump(model, f)
        joblib_size = Path('temp_joblib.pkl').stat().st_size / (1024 * 1024)  # MB
        Path('temp_joblib.pkl').unlink()  # Clean up temp file
        
        return {
            'pickle_size_mb': pickle_size,
            'joblib_size_mb': joblib_size,
            'parameter_count': self._count_parameters(model)
        }
    
    def _count_parameters(self, model) -> int:
        """Count total parameters in model"""
        
        if hasattr(model, 'estimators_'):
            # Random Forest or similar ensemble
            total_params = 0
            for estimator in model.estimators_:
                if hasattr(estimator, 'tree_'):
                    total_params += self._count_tree_parameters(estimator.tree_)
            return total_params
        elif hasattr(model, 'tree_'):
            # Single decision tree
            return self._count_tree_parameters(model.tree_)
        elif hasattr(model, 'coef_'):
            # Linear model
            return len(model.coef_.flatten()) + (1 if hasattr(model, 'intercept_') else 0)
        else:
            return 0
    
    def _count_tree_parameters(self, tree) -> int:
        """Count parameters in decision tree"""
        if hasattr(tree, 'node_count'):
            return tree.node_count * 4  # Approximate: feature, threshold, children, value
        return 0
    
    def _measure_latency(self, model, X_test: np.ndarray, 
                        n_runs: int = 100) -> Dict[str, float]:
        """Measure prediction latency"""
        
        # Warm up
        model.predict(X_test[:1])
        
        # Measure latency
        latencies = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            model.predict(X_test[:1])  # Single prediction
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'std_latency_ms': np.std(latencies)
        }
    
    def _measure_memory_usage(self, model, X_test: np.ndarray) -> Dict[str, float]:
        """Measure memory usage during prediction"""
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Load model and predict
        model.predict(X_test[:1])
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - baseline_memory
        }
    
    def _measure_accuracy(self, model, X_test: np.ndarray, 
                         y_test: np.ndarray) -> Dict[str, float]:
        """Measure model accuracy metrics"""
        
        y_pred = model.predict(X_test)
        
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    def _measure_complexity(self, model) -> Dict[str, Any]:
        """Measure model computational complexity"""
        
        complexity = {
            'model_type': type(model).__name__,
            'parameters': {}
        }
        
        if hasattr(model, 'n_estimators'):
            complexity['parameters']['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            complexity['parameters']['max_depth'] = model.max_depth
        if hasattr(model, 'min_samples_split'):
            complexity['parameters']['min_samples_split'] = model.min_samples_split
        if hasattr(model, 'min_samples_leaf'):
            complexity['parameters']['min_samples_leaf'] = model.min_samples_leaf
        
        # Estimate FLOPs for prediction
        if hasattr(model, 'estimators_'):
            complexity['estimated_flops_per_prediction'] = self._estimate_rf_flops(model)
        elif hasattr(model, 'tree_'):
            complexity['estimated_flops_per_prediction'] = self._estimate_tree_flops(model.tree_)
        
        return complexity
    
    def _estimate_rf_flops(self, rf_model) -> int:
        """Estimate FLOPs for Random Forest prediction"""
        
        if not rf_model.estimators_:
            return 0
        
        tree_flops = self._estimate_tree_flops(rf_model.estimators_[0].tree_)
        return tree_flops * len(rf_model.estimators_)
    
    def _estimate_tree_flops(self, tree) -> int:
        """Estimate FLOPs for decision tree prediction"""
        
        if hasattr(tree, 'node_count'):
            # Approximate: one comparison per node
            return tree.node_count
        return 0

class ModelCompressor:
    """Model compression and optimization"""
    
    def __init__(self):
        self.profiler = ModelProfiler()
        self.compression_history = []
        
    def compress_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       config: CompressionConfig) -> CompressionResult:
        """Compress model using specified method"""
        
        logger.info(f"Compressing model using {config.method.value}")
        
        # Profile original model
        original_profile = self.profiler.profile_model(model, X_test, y_test, "original")
        
        # Apply compression method
        if config.method == CompressionMethod.PRUNING:
            compressed_model, compression_info = self._prune_model(
                model, X_train, y_train, config
            )
        elif config.method == CompressionMethod.QUANTIZATION:
            compressed_model, compression_info = self._quantize_model(model, config)
        elif config.method == CompressionMethod.DISTILLATION:
            compressed_model, compression_info = self._distill_model(
                model, X_train, y_train, X_test, y_test, config
            )
        elif config.method == CompressionMethod.FEATURE_SELECTION:
            compressed_model, compression_info = self._select_features(
                model, X_train, y_train, X_test, y_test, config
            )
        elif config.method == CompressionMethod.ENSEMBLE_COMPRESSION:
            compressed_model, compression_info = self._compress_ensemble(
                model, X_train, y_train, config
            )
        else:
            raise ValueError(f"Unsupported compression method: {config.method}")
        
        # Profile compressed model
        compressed_profile = self.profiler.profile_model(
            compressed_model, X_test, y_test, "compressed"
        )
        
        # Calculate compression results
        result = self._calculate_compression_result(
            original_profile, compressed_profile, config, compression_info
        )
        
        # Store compression history
        self.compression_history.append(result)
        
        logger.info(f"Compression completed: ratio={result.compression_ratio:.2f}, "
                   f"accuracy_loss={result.accuracy_loss:.3f}")
        
        return result
    
    def _prune_model(self, model: RandomForestRegressor, X_train: np.ndarray,
                    y_train: np.ndarray, config: CompressionConfig) -> Tuple[Any, Dict[str, Any]]:
        """Prune model by reducing ensemble size and tree complexity"""
        
        if not hasattr(model, 'n_estimators'):
            raise ValueError("Pruning only supported for ensemble models")
        
        original_estimators = model.n_estimators
        original_depth = getattr(model, 'max_depth', None)
        
        # Calculate target ensemble size
        target_estimators = max(1, int(original_estimators * config.compression_ratio))
        
        # Create pruned model
        pruned_model = RandomForestRegressor(
            n_estimators=target_estimators,
            max_depth=min(original_depth or 10, 8),  # Reduce depth
            min_samples_split=max(model.min_samples_split, 5),
            min_samples_leaf=max(model.min_samples_leaf, 2),
            random_state=42
        )
        
        # Train pruned model
        pruned_model.fit(X_train, y_train)
        
        compression_info = {
            'original_estimators': original_estimators,
            'pruned_estimators': target_estimators,
            'original_depth': original_depth,
            'pruned_depth': pruned_model.max_depth,
            'pruning_method': 'ensemble_and_depth'
        }
        
        return pruned_model, compression_info
    
    def _quantize_model(self, model: Any, config: CompressionConfig) -> Tuple[Any, Dict[str, Any]]:
        """Quantize model parameters to reduce size"""
        
        # For tree-based models, quantization is limited
        # We'll focus on reducing precision of stored values
        
        if hasattr(model, 'estimators_'):
            # Quantize Random Forest
            quantized_model = self._quantize_random_forest(model)
        elif hasattr(model, 'tree_'):
            # Quantize Decision Tree
            quantized_model = self._quantize_decision_tree(model)
        else:
            # Return original model (no quantization possible)
            quantized_model = model
        
        compression_info = {
            'quantization_method': 'parameter_precision',
            'original_dtype': 'float64',
            'quantized_dtype': 'float32'
        }
        
        return quantized_model, compression_info
    
    def _quantize_random_forest(self, model: RandomForestRegressor) -> RandomForestRegressor:
        """Quantize Random Forest parameters"""
        
        # Create a copy with reduced precision
        quantized_model = RandomForestRegressor(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            min_samples_leaf=model.min_samples_leaf,
            random_state=model.random_state
        )
        
        # Copy estimators with quantized values
        quantized_model.estimators_ = []
        for estimator in model.estimators_:
            quantized_estimator = self._quantize_decision_tree(estimator)
            quantized_model.estimators_.append(quantized_estimator)
        
        return quantized_model
    
    def _quantize_decision_tree(self, tree: DecisionTreeRegressor) -> DecisionTreeRegressor:
        """Quantize decision tree parameters"""
        
        quantized_tree = DecisionTreeRegressor(
            max_depth=tree.max_depth,
            min_samples_split=tree.min_samples_split,
            min_samples_leaf=tree.min_samples_leaf,
            random_state=tree.random_state
        )
        
        # Copy tree structure with quantized values
        if hasattr(tree, 'tree_'):
            quantized_tree.tree_ = self._quantize_tree_structure(tree.tree_)
        
        return quantized_tree
    
    def _quantize_tree_structure(self, tree) -> Any:
        """Quantize tree structure values"""
        
        # This is a simplified quantization
        # In practice, you'd need to handle the tree structure more carefully
        quantized_tree = tree
        
        if hasattr(tree, 'value'):
            # Quantize leaf values
            quantized_tree.value = tree.value.astype(np.float32)
        
        if hasattr(tree, 'threshold'):
            # Quantize thresholds
            quantized_tree.threshold = tree.threshold.astype(np.float32)
        
        return quantized_tree
    
    def _distill_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      config: CompressionConfig) -> Tuple[Any, Dict[str, Any]]:
        """Knowledge distillation to create smaller model"""
        
        # Get teacher predictions (soft targets)
        teacher_predictions = model.predict(X_train)
        
        # Create student model (smaller architecture)
        student_model = DecisionTreeRegressor(
            max_depth=min(8, getattr(model, 'max_depth', 10) // 2),
            min_samples_split=max(5, getattr(model, 'min_samples_split', 2)),
            min_samples_leaf=max(2, getattr(model, 'min_samples_leaf', 1)),
            random_state=42
        )
        
        # Train student model with knowledge distillation
        student_model.fit(X_train, y_train)
        
        # Fine-tune with soft targets
        # This is simplified - proper distillation would use temperature scaling
        student_model.fit(X_train, teacher_predictions)
        
        compression_info = {
            'teacher_model_type': type(model).__name__,
            'student_model_type': type(student_model).__name__,
            'teacher_size': self.profiler._count_parameters(model),
            'student_size': self.profiler._count_parameters(student_model),
            'distillation_method': 'simple_regression'
        }
        
        return student_model, compression_info
    
    def _select_features(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        config: CompressionConfig) -> Tuple[Any, Dict[str, Any]]:
        """Feature selection to reduce model complexity"""
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Use permutation importance
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5)
            importances = perm_importance.importances_mean
        
        # Select top features
        n_features = len(importances)
        target_features = max(1, int(n_features * (1 - config.compression_ratio)))
        
        feature_indices = np.argsort(importances)[-target_features:]
        
        # Train model with selected features
        if hasattr(model, 'n_estimators'):
            # Random Forest
            compressed_model = RandomForestRegressor(
                n_estimators=model.n_estimators,
                max_depth=model.max_depth,
                min_samples_split=model.min_samples_split,
                min_samples_leaf=model.min_samples_leaf,
                random_state=42
            )
        else:
            # Decision Tree
            compressed_model = DecisionTreeRegressor(
                max_depth=model.max_depth,
                min_samples_split=model.min_samples_split,
                min_samples_leaf=model.min_samples_leaf,
                random_state=42
            )
        
        # Train on selected features
        compressed_model.fit(X_train[:, feature_indices], y_train)
        
        compression_info = {
            'original_features': n_features,
            'selected_features': target_features,
            'selected_indices': feature_indices.tolist(),
            'feature_importances': importances[feature_indices].tolist(),
            'selection_method': 'importance_based'
        }
        
        return compressed_model, compression_info
    
    def _compress_ensemble(self, model: RandomForestRegressor, X_train: np.ndarray,
                          y_train: np.ndarray, config: CompressionConfig) -> Tuple[Any, Dict[str, Any]]:
        """Compress ensemble using model selection and pruning"""
        
        if not hasattr(model, 'estimators_'):
            raise ValueError("Ensemble compression only supported for ensemble models")
        
        # Evaluate individual estimators
        estimator_scores = []
        for i, estimator in enumerate(model.estimators_):
            y_pred = estimator.predict(X_train)
            score = mean_squared_error(y_train, y_pred)
            estimator_scores.append((i, score))
        
        # Sort by performance (lower MSE is better)
        estimator_scores.sort(key=lambda x: x[1])
        
        # Select top estimators
        target_estimators = max(1, int(len(model.estimators_) * config.compression_ratio))
        selected_indices = [score[0] for score in estimator_scores[:target_estimators]]
        
        # Create compressed ensemble
        compressed_model = RandomForestRegressor(
            n_estimators=target_estimators,
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            min_samples_leaf=model.min_samples_leaf,
            random_state=42
        )
        
        # Copy selected estimators
        compressed_model.estimators_ = [model.estimators_[i] for i in selected_indices]
        
        compression_info = {
            'original_estimators': len(model.estimators_),
            'compressed_estimators': target_estimators,
            'selected_indices': selected_indices,
            'estimator_scores': [score[1] for score in estimator_scores[:target_estimators]],
            'compression_method': 'estimator_selection'
        }
        
        return compressed_model, compression_info
    
    def _calculate_compression_result(self, original_profile: Dict[str, Any],
                                    compressed_profile: Dict[str, Any],
                                    config: CompressionConfig,
                                    compression_info: Dict[str, Any]) -> CompressionResult:
        """Calculate compression results"""
        
        original_size = original_profile['size_metrics']['pickle_size_mb']
        compressed_size = compressed_profile['size_metrics']['pickle_size_mb']
        compression_ratio = 1 - (compressed_size / original_size)
        
        original_latency = original_profile['latency_metrics']['mean_latency_ms']
        compressed_latency = compressed_profile['latency_metrics']['mean_latency_ms']
        latency_improvement = (original_latency - compressed_latency) / original_latency
        
        original_accuracy = original_profile['accuracy_metrics']['r2']
        compressed_accuracy = compressed_profile['accuracy_metrics']['r2']
        accuracy_loss = (original_accuracy - compressed_accuracy) / original_accuracy
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            compression_ratio, latency_improvement, accuracy_loss, config
        )
        
        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            original_latency=original_latency,
            compressed_latency=compressed_latency,
            latency_improvement=latency_improvement,
            original_accuracy=original_accuracy,
            compressed_accuracy=compressed_accuracy,
            accuracy_loss=accuracy_loss,
            memory_usage={
                'original': original_profile['memory_metrics']['memory_increase_mb'],
                'compressed': compressed_profile['memory_metrics']['memory_increase_mb']
            },
            compression_method=config.method.value,
            success=accuracy_loss <= config.max_accuracy_loss,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, compression_ratio: float, latency_improvement: float,
                                accuracy_loss: float, config: CompressionConfig) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        if compression_ratio > 0.5:
            recommendations.append("Excellent compression achieved")
        elif compression_ratio > 0.3:
            recommendations.append("Good compression achieved")
        else:
            recommendations.append("Consider more aggressive compression")
        
        if latency_improvement > 0.2:
            recommendations.append("Significant latency improvement")
        elif latency_improvement > 0.1:
            recommendations.append("Moderate latency improvement")
        else:
            recommendations.append("Limited latency improvement - consider other methods")
        
        if accuracy_loss < 0.02:
            recommendations.append("Minimal accuracy loss - compression successful")
        elif accuracy_loss < 0.05:
            recommendations.append("Acceptable accuracy loss")
        else:
            recommendations.append("High accuracy loss - reduce compression intensity")
        
        # Method-specific recommendations
        if config.method == CompressionMethod.PRUNING:
            recommendations.append("Consider further tree depth reduction")
        elif config.method == CompressionMethod.FEATURE_SELECTION:
            recommendations.append("Review feature importance distribution")
        elif config.method == CompressionMethod.DISTILLATION:
            recommendations.append("Try temperature scaling for better distillation")
        
        return recommendations

class OptimizationEngine:
    """Main optimization engine for model performance"""
    
    def __init__(self):
        self.compressor = ModelCompressor()
        self.optimization_history = []
        
    def optimize_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      target: OptimizationTarget,
                      target_value: float) -> Dict[str, Any]:
        """Optimize model for specific target"""
        
        logger.info(f"Optimizing model for {target.value}: {target_value}")
        
        # Profile original model
        original_profile = self.compressor.profiler.profile_model(model, X_test, y_test, "original")
        
        # Determine best compression method based on target
        best_method = self._select_best_method(target, target_value, original_profile)
        
        # Apply compression
        config = CompressionConfig(
            method=best_method,
            target_metric=target,
            target_value=target_value,
            compression_ratio=0.5  # Default
        )
        
        result = self.compressor.compress_model(
            model, X_train, y_train, X_test, y_test, config
        )
        
        # Record optimization
        optimization_record = {
            'timestamp': time.time(),
            'target': target.value,
            'target_value': target_value,
            'method': best_method.value,
            'result': result,
            'original_profile': original_profile
        }
        
        self.optimization_history.append(optimization_record)
        
        return {
            'optimization_record': optimization_record,
            'compression_result': result,
            'best_method': best_method.value
        }
    
    def _select_best_method(self, target: OptimizationTarget, target_value: float,
                          profile: Dict[str, Any]) -> CompressionMethod:
        """Select best compression method for target"""
        
        if target == OptimizationTarget.LATENCY:
            # For latency, prefer pruning and ensemble compression
            if profile['complexity']['model_type'] == 'RandomForestRegressor':
                return CompressionMethod.ENSEMBLE_COMPRESSION
            else:
                return CompressionMethod.PRUNING
                
        elif target == OptimizationTarget.MEMORY:
            # For memory, prefer feature selection and quantization
            return CompressionMethod.FEATURE_SELECTION
            
        elif target == OptimizationTarget.SIZE:
            # For size, prefer distillation and ensemble compression
            return CompressionMethod.DISTILLATION
            
        elif target == OptimizationTarget.ACCURACY:
            # For accuracy, use gentle compression
            return CompressionMethod.FEATURE_SELECTION
            
        else:  # THROUGHPUT
            # For throughput, prefer ensemble compression
            return CompressionMethod.ENSEMBLE_COMPRESSION
    
    def benchmark_optimization(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Benchmark all optimization methods"""
        
        logger.info("Running optimization benchmark")
        
        methods = [
            CompressionMethod.PRUNING,
            CompressionMethod.QUANTIZATION,
            CompressionMethod.DISTILLATION,
            CompressionMethod.FEATURE_SELECTION,
            CompressionMethod.ENSEMBLE_COMPRESSION
        ]
        
        benchmark_results = {}
        
        for method in methods:
            try:
                config = CompressionConfig(
                    method=method,
                    target_metric=OptimizationTarget.SIZE,
                    target_value=0.5,
                    compression_ratio=0.5
                )
                
                result = self.compressor.compress_model(
                    model, X_train, y_train, X_test, y_test, config
                )
                
                benchmark_results[method.value] = result
                
            except Exception as e:
                logger.warning(f"Failed to benchmark {method.value}: {e}")
                benchmark_results[method.value] = {'error': str(e)}
        
        # Find best method
        best_method = None
        best_score = -1
        
        for method, result in benchmark_results.items():
            if 'error' not in result and result.success:
                # Composite score: compression_ratio + latency_improvement - accuracy_loss
                score = result.compression_ratio + result.latency_improvement - result.accuracy_loss
                if score > best_score:
                    best_score = score
                    best_method = method
        
        return {
            'benchmark_results': benchmark_results,
            'best_method': best_method,
            'best_score': best_score
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations"""
        
        if not self.optimization_history:
            return {'message': 'No optimizations performed'}
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'success_rate': sum(1 for record in self.optimization_history 
                              if record['result'].success) / len(self.optimization_history),
            'average_compression_ratio': np.mean([record['result'].compression_ratio 
                                               for record in self.optimization_history]),
            'average_latency_improvement': np.mean([record['result'].latency_improvement 
                                                   for record in self.optimization_history]),
            'average_accuracy_loss': np.mean([record['result'].accuracy_loss 
                                             for record in self.optimization_history]),
            'methods_used': list(set([record['method'] for record in self.optimization_history]))
        }
        
        return summary
