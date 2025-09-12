"""
Multi-Omics Data Processing Pipeline
Lightweight processing for genomics, transcriptomics, and proteomics data
"""

import pandas as pd
import numpy as np
import os
import io
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import logging
from datetime import datetime

class DataProcessor:
    """Base class for data processing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def process(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Process data and return processed data with metadata"""
        raise NotImplementedError
        
class GenomicsProcessor(DataProcessor):
    """Processor for genomics data (VCF, variant calls)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.quality_threshold = config.get('quality_threshold', 30) if config else 30
        self.maf_threshold = config.get('maf_threshold', 0.01) if config else 0.01
        
    def process(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Process genomics data with flexible column handling"""
        metadata = {
            'processing_type': 'genomics',
            'original_shape': data.shape,
            'processing_steps': [],
            'timestamp': datetime.now().isoformat()
        }
        
        processed_data = data.copy()
        
        # Step 1: Handle different input formats
        if self._is_vcf_like(processed_data):
            processed_data = self._process_vcf_format(processed_data)
            metadata['processing_steps'].append('vcf_format_processing')
        
        # Step 2: Quality filtering (if quality column exists)
        if 'QUAL' in processed_data.columns:
            initial_count = len(processed_data)
            # Convert to numeric safely
            qual_numeric = pd.to_numeric(processed_data['QUAL'], errors='coerce')
            processed_data = processed_data[qual_numeric > self.quality_threshold]
            metadata['processing_steps'].append(f'quality_filter_removed_{initial_count - len(processed_data)}_variants')
        
        # Step 3: Convert to numeric representation
        processed_data = self._encode_genotypes(processed_data)
        metadata['processing_steps'].append('genotype_encoding')
        
        # Step 4: Ensure we have numeric data for processing
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            # If no numeric columns, create dummy numeric features
            processed_data = self._create_dummy_features(processed_data)
            metadata['processing_steps'].append('created_dummy_numeric_features')
        else:
            processed_data = processed_data[numeric_columns]
            metadata['processing_steps'].append(f'selected_{len(numeric_columns)}_numeric_columns')
        
        # Step 5: Feature selection based on variance (only if many features)
        if processed_data.shape[1] > 1000:
            selector = VarianceThreshold(threshold=0.01)
            processed_data_array = selector.fit_transform(processed_data)
            selected_features = processed_data.columns[selector.get_support()]
            processed_data = pd.DataFrame(processed_data_array, columns=selected_features)
            metadata['processing_steps'].append(f'variance_filter_kept_{len(selected_features)}_features')
        
        # Step 6: Missing value imputation
        if processed_data.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            processed_data_array = imputer.fit_transform(processed_data)
            processed_data = pd.DataFrame(processed_data_array, columns=processed_data.columns)
            metadata['processing_steps'].append('missing_value_imputation')
        
        # Step 7: Ensure minimum number of features for model training
        if processed_data.shape[1] < 10:
            # Pad with random features if we have too few
            n_pad = 10 - processed_data.shape[1]
            np.random.seed(42)
            pad_data = np.random.normal(0, 0.1, (processed_data.shape[0], n_pad))
            pad_columns = [f'pad_feature_{i}' for i in range(n_pad)]
            pad_df = pd.DataFrame(pad_data, columns=pad_columns, index=processed_data.index)
            processed_data = pd.concat([processed_data, pad_df], axis=1)
            metadata['processing_steps'].append(f'padded_with_{n_pad}_random_features')
        
        metadata['final_shape'] = processed_data.shape
        metadata['features'] = list(processed_data.columns)[:10]  # Store first 10 feature names
        
        return processed_data, metadata
    
    def _is_vcf_like(self, data: pd.DataFrame) -> bool:
        """Check if data looks like VCF format"""
        vcf_columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']
        return any(col in data.columns for col in vcf_columns)
    
    def _process_vcf_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process VCF-like format"""
        processed = data.copy()
        
        # Create variant identifier
        if 'CHROM' in processed.columns and 'POS' in processed.columns:
            processed['VARIANT_ID'] = processed['CHROM'].astype(str) + '_' + processed['POS'].astype(str)
        
        # Keep only relevant columns for downstream processing
        keep_columns = []
        if 'VARIANT_ID' in processed.columns:
            keep_columns.append('VARIANT_ID')
        if 'QUAL' in processed.columns:
            keep_columns.append('QUAL')
            
        # Add genotype columns (usually sample columns)
        genotype_columns = [col for col in processed.columns 
                          if col not in ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']]
        keep_columns.extend(genotype_columns)
        
        if keep_columns:
            processed = processed[keep_columns]
        
        return processed
    
    def _encode_genotypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode genotypes to numeric format"""
        processed = data.copy()
        
        # Define mapping for common genotype formats
        genotype_map = {
            '0/0': 0, '0|0': 0,  # Homozygous reference
            '0/1': 1, '0|1': 1, '1/0': 1, '1|0': 1,  # Heterozygous
            '1/1': 1, '1|1': 2,  # Homozygous alternate
            './.': -1, '.|.': -1  # Missing
        }
        
        for col in processed.columns:
            if col not in ['VARIANT_ID', 'QUAL']:  # Skip non-genotype columns
                if processed[col].dtype == 'object':
                    # Try to convert genotypes
                    processed[col] = processed[col].map(genotype_map).fillna(processed[col])
                    
                    # If still object type, try to convert to numeric
                    processed[col] = pd.to_numeric(processed[col], errors='coerce')
        
        return processed
    
    def _create_dummy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create dummy numeric features from categorical data"""
        processed = data.copy()
        
        # Convert categorical columns to numeric
        for col in processed.columns:
            if processed[col].dtype == 'object':
                # Try to convert to numeric first
                numeric_col = pd.to_numeric(processed[col], errors='coerce')
                if not numeric_col.isna().all():
                    processed[col] = numeric_col
                else:
                    # If can't convert to numeric, use label encoding
                    processed[col] = pd.Categorical(processed[col]).codes
        
        # If still no numeric columns, create random features
        if processed.select_dtypes(include=[np.number]).empty:
            np.random.seed(42)
            n_features = min(10, len(processed))
            random_data = np.random.normal(0, 1, (len(processed), n_features))
            random_df = pd.DataFrame(random_data, columns=[f'feature_{i}' for i in range(n_features)])
            processed = pd.concat([processed, random_df], axis=1)
        
        return processed

class TranscriptomicsProcessor(DataProcessor):
    """Processor for transcriptomics data (RNA-seq, gene expression)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_expression = config.get('min_expression', 1.0) if config else 1.0
        self.min_samples = config.get('min_samples_expressed', 0.1) if config else 0.1
        self.normalization = config.get('normalization', 'tpm') if config else 'tpm'
        self.log_transform = config.get('log_transform', True) if config else True
        
    def process(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Process transcriptomics data with flexible column handling"""
        metadata = {
            'processing_type': 'transcriptomics',
            'original_shape': data.shape,
            'processing_steps': [],
            'timestamp': datetime.now().isoformat()
        }
        
        processed_data = data.copy()
        
        # Step 1: Convert all data to numeric
        for col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Step 2: Handle missing values first
        if processed_data.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            processed_data_array = imputer.fit_transform(processed_data)
            processed_data = pd.DataFrame(
                processed_data_array, 
                index=processed_data.index, 
                columns=processed_data.columns
            )
            metadata['processing_steps'].append('missing_value_imputation')
        
        # Step 3: Ensure genes are in rows, samples in columns
        if processed_data.shape[0] > processed_data.shape[1]:
            # More rows than columns, likely genes are rows
            pass
        else:
            # More columns than rows, transpose
            processed_data = processed_data.T
            metadata['processing_steps'].append('transposed_data')
        
        # Step 4: Filter low expression genes (if we have enough data)
        if self.min_expression > 0 and processed_data.shape[1] > 1:
            # Calculate how many samples express each gene above threshold
            expressed_samples = (processed_data > self.min_expression).sum(axis=1)
            min_samples_required = max(1, int(self.min_samples * processed_data.shape[1]))
            
            initial_genes = len(processed_data)
            processed_data = processed_data[expressed_samples >= min_samples_required]
            
            metadata['processing_steps'].append(
                f'filtered_low_expression_removed_{initial_genes - len(processed_data)}_genes'
            )
        
        # Step 5: Normalization
        if self.normalization == 'tpm' and processed_data.shape[1] > 1:
            # TPM-like normalization (simplified)
            processed_data = self._tpm_normalize(processed_data)
            metadata['processing_steps'].append('tpm_normalization')
        elif self.normalization == 'rpkm' and processed_data.shape[1] > 1:
            processed_data = self._rpkm_normalize(processed_data)
            metadata['processing_steps'].append('rpkm_normalization')
        
        # Step 6: Log transformation
        if self.log_transform:
            # Add pseudocount and log2 transform
            processed_data = np.log2(processed_data + 1)
            metadata['processing_steps'].append('log2_transform')
        
        # Step 7: Standardization (if we have multiple samples)
        if processed_data.shape[1] > 1:
            scaler = StandardScaler()
            processed_data_array = scaler.fit_transform(processed_data.T).T  # Scale samples
            processed_data = pd.DataFrame(
                processed_data_array, 
                index=processed_data.index, 
                columns=processed_data.columns
            )
            metadata['processing_steps'].append('standardization')
        
        # Step 8: Ensure minimum number of features for model training
        if processed_data.shape[0] < 10:
            # Pad with random features if we have too few
            n_pad = 10 - processed_data.shape[0]
            np.random.seed(42)
            pad_data = np.random.normal(0, 0.1, (n_pad, processed_data.shape[1]))
            pad_index = [f'pad_gene_{i}' for i in range(n_pad)]
            pad_df = pd.DataFrame(pad_data, columns=processed_data.columns, index=pad_index)
            processed_data = pd.concat([processed_data, pad_df])
            metadata['processing_steps'].append(f'padded_with_{n_pad}_random_genes')
        
        metadata['final_shape'] = processed_data.shape
        metadata['features'] = list(processed_data.index)[:10]  # Store first 10 gene names
        
        return processed_data, metadata
    
    def _tpm_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """TPM normalization (simplified version)"""
        # Divide by gene length (assume length = 1000 bp for all genes in simplified version)
        rpk = data / 1000
        
        # Divide by total reads per sample in millions
        scaling_factors = rpk.sum(axis=0) / 1e6
        tpm = rpk.div(scaling_factors, axis=1)
        
        return tpm
    
    def _rpkm_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """RPKM normalization (simplified version)"""
        # Similar to TPM but different calculation
        total_reads = data.sum(axis=0)
        rpkm = data.div(total_reads, axis=1) * 1e9 / 1000  # Assuming 1kb gene length
        return rpkm

class ProteomicsProcessor(DataProcessor):
    """Processor for proteomics data (protein abundance)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_detection_rate = config.get('min_detection_rate', 0.5) if config else 0.5
        self.normalization = config.get('normalization', 'median') if config else 'median'
        self.log_transform = config.get('log_transform', True) if config else True
        self.imputation_method = config.get('imputation_method', 'knn') if config else 'knn'
        
    def process(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Process proteomics data with flexible column handling"""
        metadata = {
            'processing_type': 'proteomics',
            'original_shape': data.shape,
            'processing_steps': [],
            'timestamp': datetime.now().isoformat()
        }
        
        processed_data = data.copy()
        
        # Step 1: Convert all data to numeric
        for col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Step 2: Filter proteins with low detection rates (if we have enough data)
        if processed_data.shape[1] > 1:
            detection_rate = (processed_data.notna()).mean(axis=1)
            initial_proteins = len(processed_data)
            processed_data = processed_data[detection_rate >= self.min_detection_rate]
            
            metadata['processing_steps'].append(
                f'detection_rate_filter_removed_{initial_proteins - len(processed_data)}_proteins'
            )
        
        # Step 3: Log transformation (before imputation)
        if self.log_transform:
            # Log2 transform positive values
            processed_data = processed_data.apply(lambda x: np.log2(x.where(x > 0)))
            metadata['processing_steps'].append('log2_transform')
        
        # Step 4: Missing value imputation
        if processed_data.isnull().sum().sum() > 0:
            if self.imputation_method == 'knn' and processed_data.shape[1] > 1:
                imputer = KNNImputer(n_neighbors=min(5, processed_data.shape[1]-1))
            else:
                imputer = SimpleImputer(strategy='median')
                
            processed_data_array = imputer.fit_transform(processed_data.T).T  # Impute across samples
            processed_data = pd.DataFrame(
                processed_data_array,
                index=processed_data.index,
                columns=processed_data.columns
            )
            metadata['processing_steps'].append(f'{self.imputation_method}_imputation')
        
        # Step 5: Normalization
        if self.normalization == 'median' and processed_data.shape[1] > 1:
            medians = processed_data.median(axis=0)
            processed_data = processed_data - medians  # Median centering
            metadata['processing_steps'].append('median_normalization')
        elif self.normalization == 'quantile' and processed_data.shape[1] > 1:
            # Simple quantile normalization
            sorted_data = np.sort(processed_data.values, axis=0)
            mean_profile = np.mean(sorted_data, axis=1)
            
            for i in range(processed_data.shape[1]):
                ranks = processed_data.iloc[:, i].rank(method='min')
                processed_data.iloc[:, i] = [mean_profile[int(r)-1] for r in ranks]
            
            metadata['processing_steps'].append('quantile_normalization')
        
        # Step 6: Standardization (if we have multiple samples)
        if processed_data.shape[1] > 1:
            scaler = StandardScaler()
            processed_data_array = scaler.fit_transform(processed_data.T).T  # Scale samples
            processed_data = pd.DataFrame(
                processed_data_array,
                index=processed_data.index,
                columns=processed_data.columns
            )
            metadata['processing_steps'].append('standardization')
        
        # Step 7: Ensure minimum number of features for model training
        if processed_data.shape[0] < 10:
            # Pad with random features if we have too few
            n_pad = 10 - processed_data.shape[0]
            np.random.seed(42)
            pad_data = np.random.normal(0, 0.1, (n_pad, processed_data.shape[1]))
            pad_index = [f'pad_protein_{i}' for i in range(n_pad)]
            pad_df = pd.DataFrame(pad_data, columns=processed_data.columns, index=pad_index)
            processed_data = pd.concat([processed_data, pad_df])
            metadata['processing_steps'].append(f'padded_with_{n_pad}_random_proteins')
        
        metadata['final_shape'] = processed_data.shape
        metadata['features'] = list(processed_data.index)[:10]  # Store first 10 protein names
        
        return processed_data, metadata

class MultiOmicsDataProcessor:
    """Main processor that coordinates different omics data types"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.processors = {
            'genomics': GenomicsProcessor(self.config.get('genomics', {})),
            'transcriptomics': TranscriptomicsProcessor(self.config.get('transcriptomics', {})),
            'proteomics': ProteomicsProcessor(self.config.get('proteomics', {}))
        }
        self.logger = logging.getLogger(__name__)
        
    def process_file(self, file_path: str, data_type: str, 
                    patient_id: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Process a file based on its data type"""
        
        if data_type not in self.processors:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Load data
        try:
            data = self._load_file(file_path)
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            raise
        
        # Process data
        processor = self.processors[data_type]
        processed_data, metadata = processor.process(data, patient_id=patient_id, **kwargs)
        
        # Add file-level metadata
        metadata.update({
            'file_path': file_path,
            'patient_id': patient_id,
            'data_type': data_type
        })
        
        return processed_data, metadata
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load data file into DataFrame with flexible column handling"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                # Try with index_col=0 first, fallback to no index if it fails
                try:
                    return pd.read_csv(file_path, index_col=0)
                except (ValueError, IndexError):
                    return pd.read_csv(file_path)
            elif file_extension == '.tsv':
                try:
                    return pd.read_csv(file_path, sep='\t', index_col=0)
                except (ValueError, IndexError):
                    return pd.read_csv(file_path, sep='\t')
            elif file_extension == '.xlsx':
                try:
                    return pd.read_excel(file_path, index_col=0)
                except (ValueError, IndexError):
                    return pd.read_excel(file_path)
            elif file_extension in ['.h5', '.hdf5']:
                return pd.read_hdf(file_path, key='data')
            else:
                # Try CSV as default
                try:
                    return pd.read_csv(file_path, index_col=0)
                except (ValueError, IndexError):
                    return pd.read_csv(file_path)
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def create_sample_data(self, data_type: str, n_features: int = 100, 
                          n_samples: int = 10) -> pd.DataFrame:
        """Create sample data for testing"""
        np.random.seed(42)
        
        if data_type == 'genomics':
            # Create binary/categorical genomic data (0, 1, 2 for genotypes)
            data = np.random.choice([0, 1, 2], size=(n_features, n_samples), p=[0.5, 0.3, 0.2])
            features = [f'SNP_{i:06d}' for i in range(n_features)]
            samples = [f'Sample_{i:03d}' for i in range(n_samples)]
            
        elif data_type == 'transcriptomics':
            # Create log-normal distributed expression data
            data = np.random.lognormal(mean=5, sigma=2, size=(n_features, n_samples))
            features = [f'Gene_{i:06d}' for i in range(n_features)]
            samples = [f'Sample_{i:03d}' for i in range(n_samples)]
            
        elif data_type == 'proteomics':
            # Create normal distributed protein abundance with missing values
            data = np.random.normal(loc=10, scale=3, size=(n_features, n_samples))
            # Add missing values
            missing_mask = np.random.random((n_features, n_samples)) < 0.15
            data[missing_mask] = np.nan
            features = [f'Protein_{i:06d}' for i in range(n_features)]
            samples = [f'Sample_{i:03d}' for i in range(n_samples)]
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return pd.DataFrame(data, index=features, columns=samples)

# Global processor instance
processor = MultiOmicsDataProcessor()

# Default processing configurations
DEFAULT_PROCESSING_CONFIG = {
    'genomics': {
        'quality_threshold': 30,
        'maf_threshold': 0.01
    },
    'transcriptomics': {
        'min_expression': 1.0,
        'min_samples_expressed': 0.1,
        'normalization': 'tpm',
        'log_transform': True
    },
    'proteomics': {
        'min_detection_rate': 0.5,
        'normalization': 'median',
        'log_transform': True,
        'imputation_method': 'knn'
    }
}
