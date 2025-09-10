"""
Multi-Omics Data Preprocessing Pipeline
Handles genomics, transcriptomics, and proteomics data preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scaler = None
        self.imputer = None
        self.is_fitted = False
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data"""
        pass
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize the data using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized_data = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        
        return normalized_data
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'knn') -> pd.DataFrame:
        """Handle missing values in the data"""
        if method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        elif method == 'mean':
            imputer = None
            return data.fillna(data.mean())
        elif method == 'median':
            imputer = None
            return data.fillna(data.median())
        elif method == 'drop':
            imputer = None
            return data.dropna()
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        if imputer:
            imputed_data = pd.DataFrame(
                imputer.fit_transform(data),
                index=data.index,
                columns=data.columns
            )
            return imputed_data
        
        return data

class GenomicsPreprocessor(DataPreprocessor):
    """Preprocessor for genomics data (VCF, variant calls, etc.)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.quality_threshold = config.get('quality_threshold', 30) if config else 30
        self.maf_threshold = config.get('maf_threshold', 0.01) if config else 0.01
        self.reference_genome = config.get('reference_genome', 'GRCh38') if config else 'GRCh38'
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess genomics data
        
        Args:
            data: DataFrame with variants (rows) and samples (columns)
        
        Returns:
            Preprocessed genomics data
        """
        logger.info("Starting genomics data preprocessing")
        
        # Quality filtering
        data = self._filter_by_quality(data)
        
        # Minor allele frequency filtering
        data = self._filter_by_maf(data)
        
        # Convert genotypes to numeric format
        data = self._encode_genotypes(data)
        
        # Handle missing values
        data = self.handle_missing_values(data, method='median')
        
        # Normalize if requested
        if self.config.get('normalize', True):
            data = self.normalize_data(data, method='standard')
        
        logger.info(f"Genomics preprocessing completed. Shape: {data.shape}")
        return data
    
    def _filter_by_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter variants by quality score"""
        if 'QUAL' in data.columns:
            mask = data['QUAL'] >= self.quality_threshold
            filtered_data = data.loc[mask]
            logger.info(f"Quality filtering: {len(data)} -> {len(filtered_data)} variants")
            return filtered_data
        return data
    
    def _filter_by_maf(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter variants by minor allele frequency"""
        # Calculate MAF for each variant
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            maf = data[numeric_cols].apply(lambda x: min(x.mean(), 1 - x.mean()), axis=1)
            mask = maf >= self.maf_threshold
            filtered_data = data.loc[mask]
            logger.info(f"MAF filtering: {len(data)} -> {len(filtered_data)} variants")
            return filtered_data
        return data
    
    def _encode_genotypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert genotype strings to numeric format"""
        # Common genotype encoding: 0/0 -> 0, 0/1 -> 1, 1/1 -> 2
        genotype_map = {'0/0': 0, '0|0': 0, '0/1': 1, '0|1': 1, '1/0': 1, '1|0': 1, '1/1': 2, '1|1': 2}
        
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].map(genotype_map).fillna(-1)
        
        return data

class TranscriptomicsPreprocessor(DataPreprocessor):
    """Preprocessor for transcriptomics data (RNA-seq, expression arrays)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_expression_threshold = config.get('min_expression_threshold', 1.0) if config else 1.0
        self.min_samples_expressed = config.get('min_samples_expressed', 3) if config else 3
        self.log_transform = config.get('log_transform', True) if config else True
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess transcriptomics data
        
        Args:
            data: DataFrame with genes (rows) and samples (columns)
        
        Returns:
            Preprocessed transcriptomics data
        """
        logger.info("Starting transcriptomics data preprocessing")
        
        # Filter low-expressed genes
        data = self._filter_low_expression(data)
        
        # Log transformation
        if self.log_transform:
            data = self._log_transform(data)
        
        # Normalization (TPM, FPKM, or simple scaling)
        normalization_method = self.config.get('normalization_method', 'standard')
        if normalization_method in ['standard', 'minmax', 'robust']:
            data = self.normalize_data(data, method=normalization_method)
        elif normalization_method == 'tpm':
            data = self._tpm_normalize(data)
        
        # Handle missing values
        data = self.handle_missing_values(data, method='knn')
        
        logger.info(f"Transcriptomics preprocessing completed. Shape: {data.shape}")
        return data
    
    def _filter_low_expression(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out genes with low expression"""
        # Count samples where gene expression > threshold
        expressed_samples = (data >= self.min_expression_threshold).sum(axis=1)
        mask = expressed_samples >= self.min_samples_expressed
        
        filtered_data = data.loc[mask]
        logger.info(f"Low expression filtering: {len(data)} -> {len(filtered_data)} genes")
        return filtered_data
    
    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log2 transformation with pseudocount"""
        log_data = np.log2(data + 1)
        logger.info("Applied log2 transformation")
        return log_data
    
    def _tpm_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transcripts Per Million (TPM) normalization"""
        # Simplified TPM calculation (assumes gene length information is not available)
        tpm_data = data.div(data.sum(axis=0), axis=1) * 1e6
        logger.info("Applied TPM normalization")
        return tpm_data

class ProteomicsPreprocessor(DataPreprocessor):
    """Preprocessor for proteomics data (mass spectrometry, protein arrays)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_detection_rate = config.get('min_detection_rate', 0.5) if config else 0.5
        self.batch_correction = config.get('batch_correction', False) if config else False
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess proteomics data
        
        Args:
            data: DataFrame with proteins (rows) and samples (columns)
        
        Returns:
            Preprocessed proteomics data
        """
        logger.info("Starting proteomics data preprocessing")
        
        # Filter proteins by detection rate
        data = self._filter_by_detection_rate(data)
        
        # Handle missing values (common in proteomics)
        data = self.handle_missing_values(data, method='knn')
        
        # Log transformation
        data = self._log_transform(data)
        
        # Batch correction if specified
        if self.batch_correction:
            data = self._apply_batch_correction(data)
        
        # Normalization
        normalization_method = self.config.get('normalization_method', 'median')
        if normalization_method == 'median':
            data = self._median_normalize(data)
        else:
            data = self.normalize_data(data, method=normalization_method)
        
        logger.info(f"Proteomics preprocessing completed. Shape: {data.shape}")
        return data
    
    def _filter_by_detection_rate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter proteins by detection rate across samples"""
        detection_rate = (~data.isnull()).mean(axis=1)
        mask = detection_rate >= self.min_detection_rate
        
        filtered_data = data.loc[mask]
        logger.info(f"Detection rate filtering: {len(data)} -> {len(filtered_data)} proteins")
        return filtered_data
    
    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log2 transformation"""
        # Handle zeros and negative values
        data_positive = data.clip(lower=1e-6)
        log_data = np.log2(data_positive)
        logger.info("Applied log2 transformation")
        return log_data
    
    def _median_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Median normalization for proteomics data"""
        median_values = data.median(axis=0)
        global_median = data.values.flatten()
        global_median = np.median(global_median[~np.isnan(global_median)])
        
        normalization_factors = global_median / median_values
        normalized_data = data * normalization_factors
        
        logger.info("Applied median normalization")
        return normalized_data
    
    def _apply_batch_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply batch correction (simplified implementation)"""
        # This would typically use methods like ComBat
        # For now, implement a simple mean-centering approach
        logger.warning("Batch correction not fully implemented - using mean centering")
        batch_corrected = data.subtract(data.mean(axis=1), axis=0)
        return batch_corrected

class MultiOmicsIntegrator:
    """Integrates multiple omics data types"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.dimensionality_reduction = config.get('dimensionality_reduction', 'pca') if config else 'pca'
        self.n_components = config.get('n_components', 100) if config else 100
    
    def integrate_data(self, omics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Integrate multiple omics datasets
        
        Args:
            omics_data: Dictionary with omics type as key and DataFrame as value
        
        Returns:
            Integrated multi-omics dataset
        """
        logger.info(f"Integrating {len(omics_data)} omics datasets")
        
        # Ensure all datasets have the same sample order
        common_samples = self._get_common_samples(omics_data)
        aligned_data = self._align_samples(omics_data, common_samples)
        
        # Apply dimensionality reduction to each omics type
        reduced_data = {}
        for omics_type, data in aligned_data.items():
            reduced_data[omics_type] = self._reduce_dimensions(data, omics_type)
        
        # Concatenate all omics data
        integrated_data = pd.concat(reduced_data.values(), axis=1)
        
        # Add prefixes to column names to identify omics type
        new_columns = []
        for omics_type, data in reduced_data.items():
            new_columns.extend([f"{omics_type}_{col}" for col in data.columns])
        integrated_data.columns = new_columns
        
        logger.info(f"Integration completed. Final shape: {integrated_data.shape}")
        return integrated_data
    
    def _get_common_samples(self, omics_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Find common samples across all omics datasets"""
        sample_sets = [set(data.index) for data in omics_data.values()]
        common_samples = list(set.intersection(*sample_sets))
        
        logger.info(f"Found {len(common_samples)} common samples across all omics types")
        return common_samples
    
    def _align_samples(self, omics_data: Dict[str, pd.DataFrame], common_samples: List[str]) -> Dict[str, pd.DataFrame]:
        """Align all datasets to have the same sample order"""
        aligned_data = {}
        for omics_type, data in omics_data.items():
            aligned_data[omics_type] = data.loc[common_samples].copy()
        
        return aligned_data
    
    def _reduce_dimensions(self, data: pd.DataFrame, omics_type: str) -> pd.DataFrame:
        """Apply dimensionality reduction to a single omics dataset"""
        if self.dimensionality_reduction == 'pca':
            reducer = PCA(n_components=min(self.n_components, data.shape[1]))
            reduced_data = reducer.fit_transform(data.T).T
            
            # Create column names
            columns = [f"PC{i+1}" for i in range(reduced_data.shape[1])]
            reduced_df = pd.DataFrame(reduced_data, index=data.index, columns=columns)
            
            logger.info(f"Applied PCA to {omics_type}: {data.shape} -> {reduced_df.shape}")
            return reduced_df
        
        elif self.dimensionality_reduction == 'tsne':
            # t-SNE is typically used for visualization, not preprocessing
            # But included for completeness
            reducer = TSNE(n_components=min(2, data.shape[1]), random_state=42)
            reduced_data = reducer.fit_transform(data.T).T
            
            columns = [f"tSNE{i+1}" for i in range(reduced_data.shape[1])]
            reduced_df = pd.DataFrame(reduced_data, index=data.index, columns=columns)
            
            logger.info(f"Applied t-SNE to {omics_type}: {data.shape} -> {reduced_df.shape}")
            return reduced_df
        
        else:
            # No dimensionality reduction
            logger.info(f"No dimensionality reduction applied to {omics_type}")
            return data

def create_preprocessing_pipeline(omics_type: str, config: Dict = None) -> DataPreprocessor:
    """
    Factory function to create appropriate preprocessor
    
    Args:
        omics_type: Type of omics data ('genomics', 'transcriptomics', 'proteomics')
        config: Configuration dictionary
    
    Returns:
        Appropriate preprocessor instance
    """
    preprocessors = {
        'genomics': GenomicsPreprocessor,
        'transcriptomics': TranscriptomicsPreprocessor,
        'proteomics': ProteomicsPreprocessor
    }
    
    if omics_type not in preprocessors:
        raise ValueError(f"Unknown omics type: {omics_type}")
    
    return preprocessors[omics_type](config)

def preprocess_multi_omics_data(data_dict: Dict[str, pd.DataFrame], 
                               config_dict: Dict[str, Dict] = None) -> pd.DataFrame:
    """
    Preprocess and integrate multiple omics datasets
    
    Args:
        data_dict: Dictionary with omics types as keys and DataFrames as values
        config_dict: Dictionary with omics types as keys and config dictionaries as values
    
    Returns:
        Integrated and preprocessed multi-omics dataset
    """
    config_dict = config_dict or {}
    preprocessed_data = {}
    
    # Preprocess each omics type
    for omics_type, data in data_dict.items():
        config = config_dict.get(omics_type, {})
        preprocessor = create_preprocessing_pipeline(omics_type, config)
        preprocessed_data[omics_type] = preprocessor.preprocess(data)
    
    # Integrate all omics types
    integrator_config = config_dict.get('integration', {})
    integrator = MultiOmicsIntegrator(integrator_config)
    integrated_data = integrator.integrate_data(preprocessed_data)
    
    return integrated_data
