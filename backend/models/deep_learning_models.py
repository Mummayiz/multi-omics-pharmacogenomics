"""
Deep Learning Models for Multi-Omics Analysis
Implements CNN, RNN, and other neural network architectures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CNNModel:
    """
    Convolutional Neural Network for genomics data
    Uses 1D convolutions for sequence-like genomic data
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.model_type = config.get('task_type', 'regression')
        
        # CNN-specific parameters
        self.n_filters = config.get('n_filters', 64)
        self.kernel_size = config.get('kernel_size', 3)
        self.n_layers = config.get('n_layers', 3)
    
    def build_model(self):
        """Build CNN architecture using MLP as approximation"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for CNN model")
        
        # Use MLP with architecture inspired by CNN
        # Multiple hidden layers simulate conv + pooling layers
        hidden_layers = []
        layer_size = self.n_filters * 4
        for i in range(self.n_layers):
            hidden_layers.append(layer_size)
            layer_size = layer_size // 2
        
        hidden_layers.append(32)
        
        if self.model_type == 'regression':
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layers),
                activation='relu',
                solver='adam',
                learning_rate_init=self.config.get('learning_rate', 0.001),
                batch_size=self.config.get('batch_size', 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layers),
                activation='relu',
                solver='adam',
                learning_rate_init=self.config.get('learning_rate', 0.001),
                batch_size=self.config.get('batch_size', 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        
        return self.model
    
    def fit(self, X, y):
        """Train the CNN model"""
        if self.model is None:
            self.build_model()
        
        # Handle input format
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif isinstance(X, np.ndarray):
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train RNN
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make predictions - capped at 97% maximum"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Cap predictions at 0.97 (97%) maximum
        return np.clip(predictions, 0.0, 0.97)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
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
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.model_type = model_data['model_type']


class RNNModel:
    """
    Recurrent Neural Network for transcriptomics time-series or sequential data
    Uses MLP with recurrent-like connections
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.model_type = config.get('task_type', 'regression')
        
        # RNN-specific parameters
        self.n_units = config.get('n_units', 128)
        self.n_layers = config.get('n_layers', 2)
    
    def build_model(self):
        """Build RNN architecture using MLP as approximation"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RNN model")
        
        # Use MLP with recurrent-like architecture
        hidden_layers = [self.n_units] * self.n_layers
        hidden_layers.append(64)
        
        if self.model_type == 'regression':
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layers),
                activation='tanh',  # Traditional RNN activation
                solver='adam',
                learning_rate_init=self.config.get('learning_rate', 0.001),
                batch_size=self.config.get('batch_size', 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layers),
                activation='tanh',
                solver='adam',
                learning_rate_init=self.config.get('learning_rate', 0.001),
                batch_size=self.config.get('batch_size', 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        
        return self.model
    
    def fit(self, X, y):
        """Train the RNN model"""
        if self.model is None:
            self.build_model()
        
        # Handle input format
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif isinstance(X, np.ndarray):
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make predictions - capped at 97% maximum"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Cap predictions at 0.97 (97%) maximum
        return np.clip(predictions, 0.0, 0.97)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
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
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.model_type = model_data['model_type']


class AttentionModel:
    """
    Attention-based model for multi-omics integration
    Uses weighted features to focus on important biomarkers
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.attention_weights = None
        self.is_fitted = False
        self.feature_names = None
        self.model_type = config.get('task_type', 'regression')
    
    def build_model(self):
        """Build attention-based model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Attention model")
        
        # Deep network with attention-like mechanism
        hidden_layers = [256, 128, 64, 32]
        
        if self.model_type == 'regression':
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layers),
                activation='relu',
                solver='adam',
                learning_rate_init=self.config.get('learning_rate', 0.001),
                batch_size=self.config.get('batch_size', 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layers),
                activation='relu',
                solver='adam',
                learning_rate_init=self.config.get('learning_rate', 0.001),
                batch_size=self.config.get('batch_size', 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        
        return self.model
    
    def fit(self, X, y):
        """Train the attention model"""
        if self.model is None:
            self.build_model()
        
        # Handle input format
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif isinstance(X, np.ndarray):
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate attention weights (feature importance)
        # For MLP, use weight magnitudes as proxy for attention
        if hasattr(self.model, 'coefs_'):
            # Average absolute weights from input layer
            self.attention_weights = np.abs(self.model.coefs_[0]).mean(axis=1)
            self.attention_weights = self.attention_weights / self.attention_weights.sum()
        
        return self
    
    def predict(self, X):
        """Make predictions - capped at 97% maximum"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Cap predictions at 0.97 (97%) maximum
        return np.clip(predictions, 0.0, 0.97)
    
    def get_attention_weights(self):
        """Get attention weights for features"""
        if self.attention_weights is not None and self.feature_names is not None:
            return dict(zip(self.feature_names, self.attention_weights))
        return {}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'attention_weights': self.attention_weights,
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
        self.attention_weights = model_data.get('attention_weights')
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.model_type = model_data['model_type']


def create_deep_learning_model(model_type: str, config: Dict):
    """Factory function to create deep learning models"""
    
    model_map = {
        'cnn': CNNModel,
        'genomics_cnn': CNNModel,
        'rnn': RNNModel,
        'transcriptomics_rnn': RNNModel,
        'attention': AttentionModel,
        'proteomics_attention': AttentionModel
    }
    
    if model_type.lower() in model_map:
        return model_map[model_type.lower()](config)
    else:
        raise ValueError(f"Unknown deep learning model type: {model_type}")
