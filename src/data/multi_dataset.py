"""
Multi-dataset support and transfer learning capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from ..config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
from ..utils.logger import logger

@dataclass
class DatasetMetadata:
    """Metadata for different datasets"""
    name: str
    source: str
    location: str
    time_period: str
    size: int
    features: List[str]
    target_column: str
    data_quality_score: float
    domain_characteristics: Dict[str, Any]

class DatasetRegistry:
    """Registry for managing multiple datasets"""
    
    def __init__(self):
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        
    def register_dataset(self, metadata: DatasetMetadata, data: pd.DataFrame):
        """Register a new dataset"""
        self.datasets[metadata.name] = metadata
        self.loaded_data[metadata.name] = data
        logger.info(f"Registered dataset: {metadata.name} ({len(data)} records)")
        
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get loaded dataset by name"""
        return self.loaded_data.get(name)
        
    def get_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by name"""
        return self.datasets.get(name)
        
    def list_datasets(self) -> List[str]:
        """List all registered dataset names"""
        return list(self.datasets.keys())
        
    def get_similar_datasets(self, target_name: str, 
                            similarity_metric: str = 'features') -> List[str]:
        """Find datasets similar to target dataset"""
        if target_name not in self.datasets:
            return []
            
        target_meta = self.datasets[target_name]
        similar = []
        
        for name, meta in self.datasets.items():
            if name == target_name:
                continue
                
            similarity = self._calculate_similarity(target_meta, meta, similarity_metric)
            if similarity > 0.5:  # Threshold for similarity
                similar.append((name, similarity))
                
        # Sort by similarity score
        similar.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similar]
        
    def _calculate_similarity(self, meta1: DatasetMetadata, 
                             meta2: DatasetMetadata, metric: str) -> float:
        """Calculate similarity between datasets"""
        if metric == 'features':
            # Feature overlap similarity
            features1 = set(meta1.features)
            features2 = set(meta2.features)
            intersection = len(features1.intersection(features2))
            union = len(features1.union(features2))
            return intersection / union if union > 0 else 0.0
            
        elif metric == 'location':
            # Location similarity
            return 1.0 if meta1.location == meta2.location else 0.3
            
        elif metric == 'size':
            # Size similarity (log scale)
            size_diff = abs(np.log(meta1.size) - np.log(meta2.size))
            return np.exp(-size_diff)
            
        return 0.0

class TransferLearningEngine:
    """Transfer learning for cross-dataset model adaptation"""
    
    def __init__(self):
        self.base_models = {}
        self.adapted_models = {}
        self.feature_mappers = {}
        self.scalers = {}
        
    def prepare_transfer_data(self, source_data: pd.DataFrame, 
                            target_data: pd.DataFrame,
                            source_features: List[str],
                            target_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                List[str], Dict[str, Any]]:
        """Prepare data for transfer learning"""
        
        # Find common features
        common_features = list(set(source_features) & set(target_features))
        logger.info(f"Common features for transfer: {len(common_features)}")
        
        # Align feature orders
        source_aligned = source_data[common_features].copy()
        target_aligned = target_data[common_features].copy()
        
        # Create feature mapping
        feature_mapping = {
            'common_features': common_features,
            'source_only': list(set(source_features) - set(common_features)),
            'target_only': list(set(target_features) - set(common_features)),
            'feature_types': self._get_feature_types(source_aligned)
        }
        
        return source_aligned, target_aligned, common_features, feature_mapping
        
    def _get_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Determine feature types"""
        feature_types = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                feature_types[col] = 'numerical'
            else:
                feature_types[col] = 'categorical'
        return feature_types
        
    def train_base_model(self, dataset_name: str, X: pd.DataFrame, y: pd.Series,
                        model_type: str = 'random_forest') -> Dict[str, Any]:
        """Train base model on source dataset"""
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info(f"Training base model for dataset: {dataset_name}")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        
        # Evaluate
        y_pred = model.predict(X)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'train_r2': r2_score(y, y_pred),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        # Store model
        self.base_models[dataset_name] = {
            'model': model,
            'features': X.columns.tolist(),
            'metrics': metrics
        }
        
        logger.info(f"Base model trained - RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
        
        return metrics
        
    def adapt_model(self, source_dataset: str, target_data: pd.DataFrame,
                   adaptation_method: str = 'fine_tuning') -> Dict[str, Any]:
        """Adapt base model to target dataset"""
        
        if source_dataset not in self.base_models:
            raise ValueError(f"Base model for {source_dataset} not found")
            
        base_model_info = self.base_models[source_dataset]
        base_model = base_model_info['model']
        base_features = base_model_info['features']
        
        # Prepare target data
        X_target = target_data[base_features]
        y_target = target_data[TARGET_COLUMN] if TARGET_COLUMN in target_data.columns else None
        
        logger.info(f"Adapting model from {source_dataset} using {adaptation_method}")
        
        if adaptation_method == 'fine_tuning':
            return self._fine_tune_model(base_model, X_target, y_target, source_dataset)
        elif adaptation_method == 'feature_reweighting':
            return self._reweight_features(base_model, X_target, y_target, source_dataset)
        elif adaptation_method == 'domain_adaptation':
            return self._domain_adaptation(base_model, X_target, y_target, source_dataset)
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
            
    def _fine_tune_model(self, base_model, X_target: pd.DataFrame, y_target: pd.Series,
                        source_dataset: str) -> Dict[str, Any]:
        """Fine-tune base model on target data"""
        
        # Split target data
        X_train, X_val, y_train, y_val = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )
        
        # Clone and fine-tune
        from sklearn.base import clone
        adapted_model = clone(base_model)
        
        # Fine-tune with lower learning rate (for tree-based models, use fewer estimators)
        if hasattr(adapted_model, 'n_estimators'):
            adapted_model.n_estimators = 50  # Fewer trees for fine-tuning
        if hasattr(adapted_model, 'max_depth'):
            adapted_model.max_depth = 8     # Slightly shallower
            
        adapted_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = adapted_model.predict(X_val)
        metrics = {
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'val_r2': r2_score(y_val, y_pred),
            'adaptation_method': 'fine_tuning'
        }
        
        # Store adapted model
        self.adapted_models[f"{source_dataset}_adapted"] = {
            'model': adapted_model,
            'source_dataset': source_dataset,
            'metrics': metrics
        }
        
        logger.info(f"Fine-tuned model - RMSE: {metrics['val_rmse']:.4f}, R²: {metrics['val_r2']:.4f}")
        
        return metrics
        
    def _reweight_features(self, base_model, X_target: pd.DataFrame, y_target: pd.Series,
                         source_dataset: str) -> Dict[str, Any]:
        """Rewrite feature importance based on target data"""
        
        # Calculate feature importance on target data
        from sklearn.inspection import permutation_importance
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )
        
        # Get permutation importance
        perm_importance = permutation_importance(
            base_model, X_val, y_val, n_repeats=10, random_state=42
        )
        
        # Create weighted model
        from sklearn.ensemble import RandomForestRegressor
        adapted_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Use target importance as sample weights
        feature_weights = perm_importance.importances_mean
        sample_weights = np.ones(len(X_train))
        
        # Adjust sample weights based on feature importance
        for i, feature in enumerate(X_train.columns):
            if i < len(feature_weights):
                feature_weight = feature_weights[i]
                # Apply feature weight to samples
                sample_weights *= (1 + feature_weight * X_train[feature].values / X_train[feature].std())
        
        adapted_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate
        y_pred = adapted_model.predict(X_val)
        metrics = {
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'val_r2': r2_score(y_val, y_pred),
            'adaptation_method': 'feature_reweighting',
            'feature_importance_shift': dict(zip(X_train.columns, feature_weights))
        }
        
        # Store adapted model
        self.adapted_models[f"{source_dataset}_reweighted"] = {
            'model': adapted_model,
            'source_dataset': source_dataset,
            'metrics': metrics
        }
        
        logger.info(f"Reweighted model - RMSE: {metrics['val_rmse']:.4f}, R²: {metrics['val_r2']:.4f}")
        
        return metrics
        
    def _domain_adaptation(self, base_model, X_target: pd.DataFrame, y_target: pd.Series,
                          source_dataset: str) -> Dict[str, Any]:
        """Domain adaptation using importance weighting"""
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        
        # Split target data
        X_train, X_val, y_train, y_val = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )
        
        # Get source training data (we need to store this)
        source_data = getattr(self, '_source_data', None)
        if source_data is None:
            logger.warning("No source data available for domain adaptation, using fine-tuning")
            return self._fine_tune_model(base_model, X_target, y_target, source_dataset)
        
        X_source = source_data[base_model.feature_names_in_]
        
        # Train domain classifier
        domain_labels = np.concatenate([np.ones(len(X_source)), np.zeros(len(X_train))])
        domain_X = pd.concat([X_source, X_train])
        
        domain_classifier = LogisticRegression(random_state=42)
        domain_classifier.fit(domain_X, domain_labels)
        
        # Calculate importance weights
        source_probs = domain_classifier.predict_proba(X_source)[:, 1]
        target_probs = domain_classifier.predict_proba(X_train)[:, 1]
        
        # Importance weighting
        importance_weights = target_probs / (source_probs.mean() + 1e-8)
        
        # Train adapted model with importance weights
        from sklearn.ensemble import RandomForestRegressor
        adapted_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        adapted_model.fit(X_train, y_train, sample_weight=importance_weights)
        
        # Evaluate
        y_pred = adapted_model.predict(X_val)
        metrics = {
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'val_r2': r2_score(y_val, y_pred),
            'adaptation_method': 'domain_adaptation',
            'domain_classifier_accuracy': domain_classifier.score(domain_X, domain_labels)
        }
        
        # Store adapted model
        self.adapted_models[f"{source_dataset}_domain_adapted"] = {
            'model': adapted_model,
            'source_dataset': source_dataset,
            'metrics': metrics
        }
        
        logger.info(f"Domain adapted model - RMSE: {metrics['val_rmse']:.4f}, R²: {metrics['val_r2']:.4f}")
        
        return metrics
        
    def cross_validate_transfer(self, source_datasets: List[str], 
                               target_data: pd.DataFrame) -> Dict[str, Any]:
        """Cross-validate transfer learning across multiple source datasets"""
        
        results = {}
        
        for source_dataset in source_datasets:
            if source_dataset not in self.base_models:
                continue
                
            logger.info(f"Testing transfer from {source_dataset}")
            
            # Test different adaptation methods
            methods = ['fine_tuning', 'feature_reweighting', 'domain_adaptation']
            method_results = {}
            
            for method in methods:
                try:
                    metrics = self.adapt_model(source_dataset, target_data, method)
                    method_results[method] = metrics
                except Exception as e:
                    logger.warning(f"Failed {method} adaptation from {source_dataset}: {e}")
                    method_results[method] = {'error': str(e)}
            
            results[source_dataset] = method_results
        
        # Find best performing transfer
        best_transfer = None
        best_score = float('inf')
        
        for source_dataset, methods in results.items():
            for method, metrics in methods.items():
                if 'val_rmse' in metrics and metrics['val_rmse'] < best_score:
                    best_score = metrics['val_rmse']
                    best_transfer = (source_dataset, method)
        
        return {
            'results': results,
            'best_transfer': best_transfer,
            'best_score': best_score
        }

class MultiDatasetManager:
    """Main manager for multi-dataset operations"""
    
    def __init__(self):
        self.registry = DatasetRegistry()
        self.transfer_engine = TransferLearningEngine()
        
    def load_and_register_dataset(self, file_path: str, dataset_name: str,
                                 location: str, time_period: str,
                                 source: str = "file") -> bool:
        """Load and register a new dataset"""
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality(df)
            
            # Create metadata
            metadata = DatasetMetadata(
                name=dataset_name,
                source=source,
                location=location,
                time_period=time_period,
                size=len(df),
                features=df.columns.tolist(),
                target_column=TARGET_COLUMN if TARGET_COLUMN in df.columns else df.columns[-1],
                data_quality_score=quality_score,
                domain_characteristics=self._analyze_domain_characteristics(df)
            )
            
            # Register dataset
            self.registry.register_dataset(metadata, df)
            
            logger.info(f"Successfully loaded and registered {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return False
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        score = 1.0
        
        # Missing values penalty
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 0.3
        
        # Duplicate penalty
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 0.2
        
        # Outlier penalty (using IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        total_numeric = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
            total_numeric += len(df)
        
        if total_numeric > 0:
            outlier_ratio = outlier_count / total_numeric
            score -= outlier_ratio * 0.1
        
        return max(0.0, score)
    
    def _analyze_domain_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze domain-specific characteristics"""
        characteristics = {}
        
        # Price range
        if TARGET_COLUMN in df.columns:
            characteristics['price_range'] = {
                'min': df[TARGET_COLUMN].min(),
                'max': df[TARGET_COLUMN].max(),
                'mean': df[TARGET_COLUMN].mean(),
                'std': df[TARGET_COLUMN].std()
            }
        
        # Demand-supply ratio
        if 'Number_of_Riders' in df.columns and 'Number_of_Drivers' in df.columns:
            ratio = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1e-8)
            characteristics['demand_supply_ratio'] = {
                'mean': ratio.mean(),
                'std': ratio.std(),
                'high_demand_ratio': (ratio > 1.5).mean()
            }
        
        # Location distribution
        if 'Location_Category' in df.columns:
            characteristics['location_distribution'] = df['Location_Category'].value_counts().to_dict()
        
        return characteristics
    
    def generate_synthetic_datasets(self, base_dataset: str, 
                                  variations: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Generate synthetic dataset variations for testing"""
        
        base_data = self.registry.get_dataset(base_dataset)
        if base_data is None:
            raise ValueError(f"Base dataset {base_dataset} not found")
        
        synthetic_datasets = {}
        
        for i, variation in enumerate(variations):
            synthetic_data = base_data.copy()
            
            # Apply variations
            if 'price_shift' in variation:
                if TARGET_COLUMN in synthetic_data.columns:
                    synthetic_data[TARGET_COLUMN] *= variation['price_shift']
            
            if 'demand_multiplier' in variation:
                if 'Number_of_Riders' in synthetic_data.columns:
                    synthetic_data['Number_of_Riders'] *= variation['demand_multiplier']
            
            if 'location_remap' in variation:
                if 'Location_Category' in synthetic_data.columns:
                    remap = variation['location_remap']
                    synthetic_data['Location_Category'] = synthetic_data['Location_Category'].map(remap)
            
            if 'noise_level' in variation:
                numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
                noise = np.random.normal(0, variation['noise_level'], synthetic_data[numeric_cols].shape)
                synthetic_data[numeric_cols] += noise
            
            dataset_name = f"{base_dataset}_synthetic_{i+1}"
            synthetic_datasets[dataset_name] = synthetic_data
            
            # Register synthetic dataset
            metadata = DatasetMetadata(
                name=dataset_name,
                source="synthetic",
                location="various",
                time_period="synthetic",
                size=len(synthetic_data),
                features=synthetic_data.columns.tolist(),
                target_column=TARGET_COLUMN,
                data_quality_score=self._calculate_data_quality(synthetic_data),
                domain_characteristics=self._analyze_domain_characteristics(synthetic_data)
            )
            
            self.registry.register_dataset(metadata, synthetic_data)
        
        logger.info(f"Generated {len(synthetic_datasets)} synthetic datasets")
        return synthetic_datasets
    
    def save_registry(self, file_path: str):
        """Save dataset registry to file"""
        registry_data = {
            'datasets': {
                name: {
                    'name': meta.name,
                    'source': meta.source,
                    'location': meta.location,
                    'time_period': meta.time_period,
                    'size': meta.size,
                    'features': meta.features,
                    'target_column': meta.target_column,
                    'data_quality_score': meta.data_quality_score,
                    'domain_characteristics': meta.domain_characteristics
                }
                for name, meta in self.registry.datasets.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f"Registry saved to {file_path}")
    
    def load_registry(self, file_path: str):
        """Load dataset registry from file"""
        with open(file_path, 'r') as f:
            registry_data = json.load(f)
        
        for name, data in registry_data['datasets'].items():
            metadata = DatasetMetadata(**data)
            self.registry.datasets[name] = metadata
        
        logger.info(f"Registry loaded from {file_path}")
