"""
Deep Learning Model Architectures for Multi-Omics Pharmacogenomics
Implements CNN, RNN, and Fusion models for genomics, transcriptomics, and proteomics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

class BaseMultiOmicsModel(ABC):
    """Abstract base class for multi-omics models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.is_compiled = False
    
    @abstractmethod
    def build_model(self) -> Model:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def get_model_summary(self) -> str:
        """Get model summary"""
        pass

class GenomicsCNNModel(BaseMultiOmicsModel):
    """CNN model for genomics data (variants, SNPs, etc.)"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.input_shape = config.get('input_shape', (23, 1000000, 1))  # (chromosomes, positions, channels)
        self.conv_layers = config.get('conv_layers', [64, 128, 256])
        self.kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.output_dim = config.get('output_dim', 128)
    
    def build_model(self) -> Model:
        """Build CNN model for genomics data"""
        input_layer = layers.Input(shape=self.input_shape, name='genomics_input')
        
        x = input_layer
        
        # Multi-scale convolutional layers
        conv_outputs = []
        for i, (filters, kernel_size) in enumerate(zip(self.conv_layers, self.kernel_sizes)):
            # Conv block with different kernel sizes
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                name=f'conv1d_{i+1}_k{kernel_size}'
            )(x)
            
            # Batch normalization
            conv = layers.BatchNormalization(name=f'bn_{i+1}_k{kernel_size}')(conv)
            
            # Max pooling
            conv = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}_k{kernel_size}')(conv)
            
            # Dropout
            conv = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}_k{kernel_size}')(conv)
            
            conv_outputs.append(conv)
        
        # Concatenate multi-scale features
        if len(conv_outputs) > 1:
            x = layers.Concatenate(name='multi_scale_concat')(conv_outputs)
        else:
            x = conv_outputs[0]
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout')(x)
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        
        # Output layer
        output = layers.Dense(self.output_dim, activation='relu', name='genomics_output')(x)
        
        model = Model(inputs=input_layer, outputs=output, name='GenomicsCNN')
        return model
    
    def get_model_summary(self) -> str:
        if self.model is None:
            self.model = self.build_model()
        return str(self.model.summary())

class TranscriptomicsRNNModel(BaseMultiOmicsModel):
    """RNN/LSTM model for transcriptomics data (gene expression time series)"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.input_shape = config.get('input_shape', (None, 20000))  # (time_points, genes)
        self.rnn_units = config.get('rnn_units', [128, 64])
        self.rnn_type = config.get('rnn_type', 'LSTM')  # LSTM, GRU
        self.dropout_rate = config.get('dropout_rate', 0.4)
        self.output_dim = config.get('output_dim', 128)
        self.bidirectional = config.get('bidirectional', True)
    
    def build_model(self) -> Model:
        """Build RNN model for transcriptomics data"""
        input_layer = layers.Input(shape=self.input_shape, name='transcriptomics_input')
        
        x = input_layer
        
        # RNN layers
        for i, units in enumerate(self.rnn_units):
            return_sequences = i < len(self.rnn_units) - 1
            
            if self.rnn_type == 'LSTM':
                rnn_layer = layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate,
                    name=f'lstm_{i+1}'
                )
            elif self.rnn_type == 'GRU':
                rnn_layer = layers.GRU(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate,
                    name=f'gru_{i+1}'
                )
            else:
                raise ValueError(f"Unknown RNN type: {self.rnn_type}")
            
            # Apply bidirectional wrapper if requested
            if self.bidirectional:
                rnn_layer = layers.Bidirectional(rnn_layer, name=f'bidirectional_{i+1}')
            
            x = rnn_layer(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        
        # Output layer
        output = layers.Dense(self.output_dim, activation='relu', name='transcriptomics_output')(x)
        
        model = Model(inputs=input_layer, outputs=output, name='TranscriptomicsRNN')
        return model
    
    def get_model_summary(self) -> str:
        if self.model is None:
            self.model = self.build_model()
        return str(self.model.summary())

class ProteomicsFCModel(BaseMultiOmicsModel):
    """Fully connected model for proteomics data"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.input_shape = config.get('input_shape', (10000,))  # number of proteins
        self.hidden_layers = config.get('hidden_layers', [512, 256, 128])
        self.activation = config.get('activation', 'relu')
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.output_dim = config.get('output_dim', 128)
        self.batch_norm = config.get('batch_norm', True)
    
    def build_model(self) -> Model:
        """Build fully connected model for proteomics data"""
        input_layer = layers.Input(shape=self.input_shape, name='proteomics_input')
        
        x = input_layer
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, name=f'dense_{i+1}')(x)
            
            if self.batch_norm:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            x = layers.Activation(self.activation, name=f'activation_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        output = layers.Dense(self.output_dim, activation='relu', name='proteomics_output')(x)
        
        model = Model(inputs=input_layer, outputs=output, name='ProteomicsFC')
        return model
    
    def get_model_summary(self) -> str:
        if self.model is None:
            self.model = self.build_model()
        return str(self.model.summary())

class AttentionLayer(layers.Layer):
    """Custom attention layer for multi-omics fusion"""
    
    def __init__(self, attention_dim: int = 64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.W = None
        self.b = None
        self.u = None
    
    def build(self, input_shape):
        # Weight matrix
        self.W = self.add_weight(
            name='attention_W',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Bias vector
        self.b = self.add_weight(
            name='attention_b',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # Context vector
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.attention_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, seq_len, features)
        # Apply attention mechanism
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.squeeze(ait, -1)
        
        # Apply softmax to get attention weights
        ait = tf.nn.softmax(ait, axis=-1)
        
        # Apply weights to inputs
        weighted_input = inputs * tf.expand_dims(ait, -1)
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output, ait
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])

class MultiOmicsFusionModel(BaseMultiOmicsModel):
    """Multi-branch fusion model combining genomics, transcriptomics, and proteomics"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.genomics_config = config.get('genomics', {})
        self.transcriptomics_config = config.get('transcriptomics', {})
        self.proteomics_config = config.get('proteomics', {})
        
        self.attention_heads = config.get('attention_heads', 8)
        self.attention_dim = config.get('attention_dim', 64)
        self.fusion_type = config.get('fusion_type', 'late')  # early, late, intermediate
        self.final_layers = config.get('final_layers', [256, 128, 64])
        self.num_classes = config.get('num_classes', 1)  # For drug response prediction
        self.dropout_rate = config.get('dropout_rate', 0.3)
    
    def build_model(self) -> Model:
        """Build multi-omics fusion model"""
        # Individual branch models
        genomics_model = GenomicsCNNModel(self.genomics_config).build_model()
        transcriptomics_model = TranscriptomicsRNNModel(self.transcriptomics_config).build_model()
        proteomics_model = ProteomicsFCModel(self.proteomics_config).build_model()
        
        # Extract feature layers (remove final output layers)
        genomics_features = genomics_model.layers[-2].output  # Before final dense layer
        transcriptomics_features = transcriptomics_model.layers[-2].output
        proteomics_features = proteomics_model.layers[-2].output
        
        # Fusion strategy
        if self.fusion_type == 'early':
            # Concatenate features early and process together
            fused_features = layers.Concatenate(name='early_fusion')([
                genomics_features, transcriptomics_features, proteomics_features
            ])
            
        elif self.fusion_type == 'late':
            # Process each omics type separately, then combine
            genomics_processed = layers.Dense(128, activation='relu', name='genomics_proc')(genomics_features)
            transcriptomics_processed = layers.Dense(128, activation='relu', name='transcriptomics_proc')(transcriptomics_features)
            proteomics_processed = layers.Dense(128, activation='relu', name='proteomics_proc')(proteomics_features)
            
            fused_features = layers.Concatenate(name='late_fusion')([
                genomics_processed, transcriptomics_processed, proteomics_processed
            ])
            
        elif self.fusion_type == 'attention':
            # Use attention mechanism for fusion
            # Stack features for attention
            stacked_features = tf.stack([genomics_features, transcriptomics_features, proteomics_features], axis=1)
            
            # Apply attention
            attention_layer = AttentionLayer(self.attention_dim, name='attention_fusion')
            fused_features, attention_weights = attention_layer(stacked_features)
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Final processing layers
        x = fused_features
        for i, units in enumerate(self.final_layers):
            x = layers.Dense(units, activation='relu', name=f'final_dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'final_dropout_{i+1}')(x)
        
        # Output layer for drug response prediction
        if self.num_classes == 1:
            # Regression (continuous drug response)
            output = layers.Dense(1, activation='sigmoid', name='drug_response_output')(x)
        else:
            # Classification (response categories)
            output = layers.Dense(self.num_classes, activation='softmax', name='drug_response_output')(x)
        
        # Create the full model
        inputs = [
            genomics_model.input,
            transcriptomics_model.input,
            proteomics_model.input
        ]
        
        model = Model(inputs=inputs, outputs=output, name='MultiOmicsFusion')
        return model
    
    def get_model_summary(self) -> str:
        if self.model is None:
            self.model = self.build_model()
        return str(self.model.summary())

class DrugResponsePredictor:
    """Main predictor class that orchestrates multi-omics models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config.get('model_type', 'multi_omics_fusion')
        self.models = {}
        self.is_trained = False
    
    def build_models(self):
        """Build all required models"""
        if self.model_type == 'genomics_cnn':
            self.models['genomics'] = GenomicsCNNModel(self.config.get('genomics', {}))
        elif self.model_type == 'transcriptomics_rnn':
            self.models['transcriptomics'] = TranscriptomicsRNNModel(self.config.get('transcriptomics', {}))
        elif self.model_type == 'proteomics_fc':
            self.models['proteomics'] = ProteomicsFCModel(self.config.get('proteomics', {}))
        elif self.model_type == 'multi_omics_fusion':
            self.models['fusion'] = MultiOmicsFusionModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def compile_models(self, optimizer: str = 'adam', learning_rate: float = 0.001):
        """Compile models with optimizer and loss function"""
        if self.config.get('num_classes', 1) == 1:
            # Regression
            loss = 'mse'
            metrics = ['mae', 'mse']
        else:
            # Classification
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        for model_name, model_obj in self.models.items():
            if model_obj.model is None:
                model_obj.model = model_obj.build_model()
            
            model_obj.model.compile(
                optimizer=opt,
                loss=loss,
                metrics=metrics
            )
            model_obj.is_compiled = True
    
    def train_models(self, train_data: Dict, val_data: Dict = None, 
                    epochs: int = 100, batch_size: int = 32, **kwargs):
        """Train the models"""
        # Training implementation would go here
        # This is a placeholder for the actual training logic
        pass
    
    def predict(self, data: Dict) -> Dict:
        """Make predictions using the trained models"""
        predictions = {}
        
        for model_name, model_obj in self.models.items():
            if model_obj.model is not None and model_obj.is_compiled:
                # Extract relevant data for this model
                model_data = self._extract_model_data(data, model_name)
                pred = model_obj.model.predict(model_data)
                predictions[model_name] = pred
        
        return predictions
    
    def _extract_model_data(self, data: Dict, model_name: str):
        """Extract relevant data for a specific model"""
        if model_name == 'genomics' or 'genomics' in model_name:
            return data.get('genomics')
        elif model_name == 'transcriptomics' or 'transcriptomics' in model_name:
            return data.get('transcriptomics')
        elif model_name == 'proteomics' or 'proteomics' in model_name:
            return data.get('proteomics')
        elif model_name == 'fusion':
            return [data.get('genomics'), data.get('transcriptomics'), data.get('proteomics')]
        else:
            return data

def create_model_architecture(model_type: str, config: Dict) -> BaseMultiOmicsModel:
    """
    Factory function to create model architectures
    
    Args:
        model_type: Type of model to create
        config: Model configuration dictionary
    
    Returns:
        Model instance
    """
    model_classes = {
        'genomics_cnn': GenomicsCNNModel,
        'transcriptomics_rnn': TranscriptomicsRNNModel,
        'proteomics_fc': ProteomicsFCModel,
        'multi_omics_fusion': MultiOmicsFusionModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](config)
