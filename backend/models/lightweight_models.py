"""
Multi-Omics Pharmacogenomics Platform - Lightweight ML Models
Using scikit-learn for easy deployment and fast inference
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

class BaseMultiOmicsModel(ABC):
    """Abstract base class for multi-omics models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.is_fitted = False
        self.feature_names = None
        self.model_type = config.get('task_type', 'regression')  # regression or classification
    
    @abstractmethod
    def build_model(self):
        """Build the model pipeline"""
        pass
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance if available"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.model.coef_).flatten()))
        else:
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.model_type = model_data['model_type']

class GenomicsModel(BaseMultiOmicsModel):
    """Genomics model using ensemble methods for SNP/variant data"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.n_features = config.get('max_features', 10000)
        self.algorithm = config.get('algorithm', 'random_forest')
        
    def build_model(self):
        """Build genomics model pipeline"""
        # Feature selection
        if self.model_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=min(self.n_features, 5000))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(self.n_features, 5000))
        
        # Scaler
        scaler = StandardScaler()
        
        # Model selection
        if self.algorithm == 'random_forest':
            if self.model_type == 'regression':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
        elif self.algorithm in ['xgboost', 'lightgbm']:
            # Fallback to RandomForest when gradient boosting libs are unavailable
            if self.model_type == 'regression':
                model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1
                )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('selector', selector),
            ('scaler', scaler),
            ('model', model)
        ])
        
        return self.pipeline
    
    def fit(self, X, y):
        """Fit the genomics model"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            self.build_model()
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            self.feature_names = [f'SNP_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            self.feature_names = list(X.columns)
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Extract components
        self.feature_selector = self.pipeline.named_steps['selector']
        self.scaler = self.pipeline.named_steps['scaler']
        self.model = self.pipeline.named_steps['model']
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities (for classification)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model_type == 'classification':
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)
            return self.pipeline.predict_proba(X)
        else:
            raise ValueError("predict_proba only available for classification tasks")

class TranscriptomicsModel(BaseMultiOmicsModel):
    """Transcriptomics model for gene expression data"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.n_components = config.get('n_components', 100)
        self.algorithm = config.get('algorithm', 'elastic_net')
        self.use_pca = config.get('use_pca', True)
        
    def build_model(self):
        """Build transcriptomics model pipeline"""
        pipeline_steps = []
        
        # Scaling
        pipeline_steps.append(('scaler', StandardScaler()))
        
        # Dimensionality reduction
        if self.use_pca:
            pipeline_steps.append(('pca', PCA(n_components=self.n_components)))
        
        # Model selection
        if self.algorithm == 'elastic_net':
            if self.model_type == 'regression':
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            else:
                model = LogisticRegression(penalty='elasticnet', solver='saga', 
                                         l1_ratio=0.5, random_state=42, max_iter=1000)
        elif self.algorithm == 'random_forest':
            if self.model_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.algorithm == 'gradient_boosting':
            if self.model_type == 'regression':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.algorithm == 'neural_network':
            if self.model_type == 'regression':
                model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            else:
                model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        pipeline_steps.append(('model', model))
        
        # Create pipeline
        self.pipeline = Pipeline(pipeline_steps)
        return self.pipeline
    
    def fit(self, X, y):
        """Fit the transcriptomics model"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            self.build_model()
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            self.feature_names = [f'Gene_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            self.feature_names = list(X.columns)
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Extract components
        self.scaler = self.pipeline.named_steps['scaler']
        self.model = self.pipeline.named_steps['model']
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        return self.pipeline.predict(X)

class ProteomicsModel(BaseMultiOmicsModel):
    """Proteomics model for protein abundance data"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.algorithm = config.get('algorithm', 'svm')
        self.kernel = config.get('kernel', 'rbf')
        
    def build_model(self):
        """Build proteomics model pipeline"""
        # Scaling
        scaler = MinMaxScaler()
        
        # Model selection
        if self.algorithm == 'svm':
            if self.model_type == 'regression':
                model = SVR(kernel=self.kernel, C=1.0, gamma='scale')
            else:
                model = SVC(kernel=self.kernel, C=1.0, gamma='scale', probability=True)
        elif self.algorithm == 'ridge':
            if self.model_type == 'regression':
                model = Ridge(alpha=1.0, random_state=42)
            else:
                model = LogisticRegression(penalty='l2', random_state=42)
        elif self.algorithm == 'random_forest':
            if self.model_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        return self.pipeline
    
    def fit(self, X, y):
        """Fit the proteomics model"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            self.build_model()
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            self.feature_names = [f'Protein_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            self.feature_names = list(X.columns)
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Extract components
        self.scaler = self.pipeline.named_steps['scaler']
        self.model = self.pipeline.named_steps['model']
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        return self.pipeline.predict(X)

class MultiOmicsFusionModel:
    """Multi-omics fusion model combining all omics types"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.genomics_model = None
        self.transcriptomics_model = None
        self.proteomics_model = None
        self.fusion_model = None
        self.is_fitted = False
        self.model_type = config.get('task_type', 'regression')
        self.fusion_strategy = config.get('fusion_strategy', 'concatenation')  # concatenation, voting, stacking
        
    def build_models(self):
        """Build all individual omics models"""
        # Individual models
        genomics_config = self.config.get('genomics', {})
        genomics_config['task_type'] = self.model_type
        self.genomics_model = GenomicsModel(genomics_config)
        
        transcriptomics_config = self.config.get('transcriptomics', {})
        transcriptomics_config['task_type'] = self.model_type
        self.transcriptomics_model = TranscriptomicsModel(transcriptomics_config)
        
        proteomics_config = self.config.get('proteomics', {})
        proteomics_config['task_type'] = self.model_type
        self.proteomics_model = ProteomicsModel(proteomics_config)
        
        # Fusion model
        if self.fusion_strategy == 'concatenation':
            if self.model_type == 'regression':
                self.fusion_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            else:
                self.fusion_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        elif self.fusion_strategy == 'stacking':
            # Use a meta-learner for stacking
            if self.model_type == 'regression':
                self.fusion_model = Ridge(alpha=1.0, random_state=42)
            else:
                self.fusion_model = LogisticRegression(random_state=42)
        
        return self
    
    def fit(self, X_dict: Dict, y):
        """Fit all models"""
        if self.genomics_model is None:
            self.build_models()
        
        # Fit individual models
        if 'genomics' in X_dict:
            self.genomics_model.fit(X_dict['genomics'], y)
        if 'transcriptomics' in X_dict:
            self.transcriptomics_model.fit(X_dict['transcriptomics'], y)
        if 'proteomics' in X_dict:
            self.proteomics_model.fit(X_dict['proteomics'], y)
        
        # Create fusion features
        fusion_features = self._create_fusion_features(X_dict)
        
        # Fit fusion model
        if fusion_features.shape[1] > 0:
            self.fusion_model.fit(fusion_features, y)
            self.is_fitted = True
        
        return self
    
    def predict(self, X_dict: Dict):
        """Make predictions using the fusion model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create fusion features
        fusion_features = self._create_fusion_features(X_dict)
        
        if fusion_features.shape[1] > 0:
            return self.fusion_model.predict(fusion_features)
        else:
            raise ValueError("No valid features for prediction")
    
    def _create_fusion_features(self, X_dict: Dict) -> np.ndarray:
        """Create fusion features from individual model predictions"""
        features = []
        
        if self.fusion_strategy == 'concatenation':
            # Concatenate original features (dimensionality reduced)
            if 'genomics' in X_dict and self.genomics_model.is_fitted:
                try:
                    # Use feature selection to reduce dimensionality
                    genomics_features = self.genomics_model.pipeline.named_steps['selector'].transform(
                        self.genomics_model.pipeline.named_steps['scaler'].transform(X_dict['genomics'])
                    )
                    # Take up to 100 features, or all if fewer
                    n_features = min(100, genomics_features.shape[1])
                    features.append(genomics_features[:, :n_features])
                except Exception as e:
                    # Fallback: use raw features with padding/truncation
                    genomics_features = self.genomics_model.pipeline.named_steps['scaler'].transform(X_dict['genomics'])
                    n_features = min(100, genomics_features.shape[1])
                    if genomics_features.shape[1] < 100:
                        # Pad with zeros
                        pad_width = 100 - genomics_features.shape[1]
                        padding = np.zeros((genomics_features.shape[0], pad_width))
                        genomics_features = np.hstack([genomics_features, padding])
                    features.append(genomics_features[:, :100])
            
            if 'transcriptomics' in X_dict and self.transcriptomics_model.is_fitted:
                try:
                    transcriptomics_features = self.transcriptomics_model.pipeline.named_steps['scaler'].transform(
                        X_dict['transcriptomics']
                    )
                    if 'pca' in self.transcriptomics_model.pipeline.named_steps:
                        transcriptomics_features = self.transcriptomics_model.pipeline.named_steps['pca'].transform(
                            transcriptomics_features
                        )
                    # Ensure consistent feature count
                    n_features = min(100, transcriptomics_features.shape[1])
                    if transcriptomics_features.shape[1] < 100:
                        pad_width = 100 - transcriptomics_features.shape[1]
                        padding = np.zeros((transcriptomics_features.shape[0], pad_width))
                        transcriptomics_features = np.hstack([transcriptomics_features, padding])
                    features.append(transcriptomics_features[:, :100])
                except Exception as e:
                    # Fallback: use raw features
                    transcriptomics_features = self.transcriptomics_model.pipeline.named_steps['scaler'].transform(
                        X_dict['transcriptomics']
                    )
                    n_features = min(100, transcriptomics_features.shape[1])
                    if transcriptomics_features.shape[1] < 100:
                        pad_width = 100 - transcriptomics_features.shape[1]
                        padding = np.zeros((transcriptomics_features.shape[0], pad_width))
                        transcriptomics_features = np.hstack([transcriptomics_features, padding])
                    features.append(transcriptomics_features[:, :100])
            
            if 'proteomics' in X_dict and self.proteomics_model.is_fitted:
                try:
                    proteomics_features = self.proteomics_model.pipeline.named_steps['scaler'].transform(
                        X_dict['proteomics']
                    )
                    # Ensure consistent feature count
                    n_features = min(100, proteomics_features.shape[1])
                    if proteomics_features.shape[1] < 100:
                        pad_width = 100 - proteomics_features.shape[1]
                        padding = np.zeros((proteomics_features.shape[0], pad_width))
                        proteomics_features = np.hstack([proteomics_features, padding])
                    features.append(proteomics_features[:, :100])
                except Exception as e:
                    # Fallback: use raw features
                    proteomics_features = self.proteomics_model.pipeline.named_steps['scaler'].transform(
                        X_dict['proteomics']
                    )
                    n_features = min(100, proteomics_features.shape[1])
                    if proteomics_features.shape[1] < 100:
                        pad_width = 100 - proteomics_features.shape[1]
                        padding = np.zeros((proteomics_features.shape[0], pad_width))
                        proteomics_features = np.hstack([proteomics_features, padding])
                    features.append(proteomics_features[:, :100])
        
        elif self.fusion_strategy == 'stacking':
            # Use predictions from individual models as features
            if 'genomics' in X_dict and self.genomics_model.is_fitted:
                genomics_pred = self.genomics_model.predict(X_dict['genomics']).reshape(-1, 1)
                features.append(genomics_pred)
            
            if 'transcriptomics' in X_dict and self.transcriptomics_model.is_fitted:
                transcriptomics_pred = self.transcriptomics_model.predict(X_dict['transcriptomics']).reshape(-1, 1)
                features.append(transcriptomics_pred)
            
            if 'proteomics' in X_dict and self.proteomics_model.is_fitted:
                proteomics_pred = self.proteomics_model.predict(X_dict['proteomics']).reshape(-1, 1)
                features.append(proteomics_pred)
        
        if features:
            return np.hstack(features)
        else:
            return np.array([]).reshape(len(list(X_dict.values())[0]), 0)
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from all models"""
        importance_dict = {}
        
        if hasattr(self.genomics_model, 'get_feature_importance') and self.genomics_model.is_fitted:
            importance_dict['genomics'] = self.genomics_model.get_feature_importance()
        
        if hasattr(self.transcriptomics_model, 'get_feature_importance') and self.transcriptomics_model.is_fitted:
            importance_dict['transcriptomics'] = self.transcriptomics_model.get_feature_importance()
        
        if hasattr(self.proteomics_model, 'get_feature_importance') and self.proteomics_model.is_fitted:
            importance_dict['proteomics'] = self.proteomics_model.get_feature_importance()
        
        if hasattr(self.fusion_model, 'feature_importances_'):
            importance_dict['fusion'] = dict(enumerate(self.fusion_model.feature_importances_))
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save the entire fusion model"""
        model_data = {
            'genomics_model': self.genomics_model,
            'transcriptomics_model': self.transcriptomics_model,
            'proteomics_model': self.proteomics_model,
            'fusion_model': self.fusion_model,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'model_type': self.model_type,
            'fusion_strategy': self.fusion_strategy
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load the entire fusion model"""
        model_data = joblib.load(filepath)
        self.genomics_model = model_data['genomics_model']
        self.transcriptomics_model = model_data['transcriptomics_model']
        self.proteomics_model = model_data['proteomics_model']
        self.fusion_model = model_data['fusion_model']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.model_type = model_data['model_type']
        self.fusion_strategy = model_data['fusion_strategy']

# Model Factory Function
def create_lightweight_model(model_type: str, config: Dict):
    """Factory function to create lightweight models"""
    
    if model_type == 'genomics':
        return GenomicsModel(config)
    elif model_type == 'transcriptomics':
        return TranscriptomicsModel(config)
    elif model_type == 'proteomics':
        return ProteomicsModel(config)
    elif model_type == 'multi_omics_fusion':
        return MultiOmicsFusionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example configuration
DEFAULT_CONFIG = {
    'task_type': 'regression',  # or 'classification'
    'genomics': {
        'algorithm': 'random_forest',  # random_forest, xgboost, lightgbm
        'max_features': 10000
    },
    'transcriptomics': {
        'algorithm': 'elastic_net',  # elastic_net, random_forest, gradient_boosting, neural_network
        'n_components': 100,
        'use_pca': True
    },
    'proteomics': {
        'algorithm': 'svm',  # svm, ridge, random_forest
        'kernel': 'rbf'
    },
    'fusion_strategy': 'concatenation'  # concatenation, stacking
}
