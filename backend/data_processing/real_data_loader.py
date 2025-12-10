"""
Real Dataset Loader for Multi-Omics Platform
Loads and processes the comprehensive 50-patient multi-omics dataset
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RealDatasetLoader:
    """Loads real patient data from comprehensive multi-omics CSV file"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.main_dataset_file = "pharmacogenomics_multiomics_50patients.csv"
        
        # Cache for loaded data
        self.main_data = None
        self.patient_ids = []
        
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load the comprehensive multi-omics dataset"""
        try:
            # Load main dataset
            main_path = os.path.join(self.data_dir, self.main_dataset_file)
            if os.path.exists(main_path):
                self.main_data = pd.read_csv(main_path)
                self.patient_ids = self.main_data['patient_id'].unique().tolist()
                logger.info(f"✓ Loaded comprehensive dataset: {self.main_data.shape}")
                logger.info(f"✓ Found {len(self.patient_ids)} patients")
            else:
                logger.warning(f"Main dataset file not found: {main_path}")
                
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
    
    def get_available_patients(self) -> List[str]:
        """Get list of all available patient IDs"""
        if self.main_data is not None:
            return self.patient_ids
        return []
    
    def patient_exists(self, patient_id: str) -> bool:
        """Check if a patient ID exists in any dataset"""
        return patient_id in self.get_available_patients()
    
    def get_patient_genomics(self, patient_id: str) -> Optional[pd.DataFrame]:
        """Get genomics data for a specific patient"""
        if self.main_data is None:
            return None
        
        patient_data = self.main_data[self.main_data['patient_id'] == patient_id]
        
        if len(patient_data) == 0:
            logger.warning(f"No genomics data found for patient {patient_id}")
            return None
        
        # Extract genomics-related columns
        genomics_cols = ['patient_id', 'CYP2D6_metabolizer_status', 'CYP2C19_metabolizer_status',
                        'ABCB1_variant_allele', 'HLA_B_1502_present', 'VKORC1_variant_allele']
        available_cols = [col for col in genomics_cols if col in patient_data.columns]
        
        genomics_data = patient_data[available_cols]
        logger.info(f"Retrieved genomics data for {patient_id}: {genomics_data.shape}")
        return genomics_data
    
    def get_patient_transcriptomics(self, patient_id: str) -> Optional[pd.DataFrame]:
        """Get transcriptomics data for a specific patient"""
        if self.main_data is None:
            return None
        
        patient_data = self.main_data[self.main_data['patient_id'] == patient_id]
        
        if len(patient_data) == 0:
            logger.warning(f"No transcriptomics data found for patient {patient_id}")
            return None
        
        # Extract RNA/transcriptomics-related columns
        transcriptomics_cols = ['patient_id', 'CYP2D6_RNA_log2TPM', 'CYP2C19_RNA_log2TPM',
                               'ABCB1_RNA_log2TPM', 'IL6_RNA_log2TPM', 'TNF_RNA_log2TPM']
        available_cols = [col for col in transcriptomics_cols if col in patient_data.columns]
        
        transcriptomics_data = patient_data[available_cols]
        logger.info(f"Retrieved transcriptomics data for {patient_id}: {transcriptomics_data.shape}")
        return transcriptomics_data
    
    def get_patient_proteomics(self, patient_id: str) -> Optional[pd.DataFrame]:
        """Get proteomics data for a specific patient"""
        if self.main_data is None:
            return None
        
        patient_data = self.main_data[self.main_data['patient_id'] == patient_id]
        
        if len(patient_data) == 0:
            logger.warning(f"No proteomics data found for patient {patient_id}")
            return None
        
        # Extract protein-related columns
        proteomics_cols = ['patient_id', 'CYP2D6_protein_log2', 'CYP2C19_protein_log2',
                          'ABCB1_protein_log2', 'Albumin_protein_log2', 'CRP_protein_log2']
        available_cols = [col for col in proteomics_cols if col in patient_data.columns]
        
        proteomics_data = patient_data[available_cols]
        logger.info(f"Retrieved proteomics data for {patient_id}: {proteomics_data.shape}")
        return proteomics_data
    
    def get_patient_drug_response(self, patient_id: str, drug_name: str = None) -> Optional[float]:
        """Get actual drug response for a patient"""
        if self.main_data is None:
            return None
        
        patient_data = self.main_data[self.main_data['patient_id'] == patient_id]
        
        if len(patient_data) == 0:
            logger.warning(f"No drug response data found for patient {patient_id}")
            return None
        
        # Get DrugX response score (normalized between 0 and 1)
        if 'DrugX_response_score' in patient_data.columns:
            response = patient_data['DrugX_response_score'].iloc[0]
            return float(response)
        
        return None
    
    def get_patient_response_class(self, patient_id: str) -> Optional[str]:
        """Get response classification (Responder/Non-responder)"""
        if self.main_data is None:
            return None
        
        patient_data = self.main_data[self.main_data['patient_id'] == patient_id]
        
        if len(patient_data) == 0 or 'DrugX_response_class' not in patient_data.columns:
            return None
        
        return patient_data['DrugX_response_class'].iloc[0]
    
    def get_patient_features(self, patient_id: str, data_type: str) -> Optional[np.ndarray]:
        """
        Get patient features as numpy array for model input
        Excludes patient_id column and converts to numeric features
        """
        if data_type == 'genomics':
            data = self.get_patient_genomics(patient_id)
            feature_cols = ['CYP2D6_metabolizer_status', 'CYP2C19_metabolizer_status',
                           'ABCB1_variant_allele', 'HLA_B_1502_present', 'VKORC1_variant_allele']
        elif data_type == 'transcriptomics':
            data = self.get_patient_transcriptomics(patient_id)
            feature_cols = ['CYP2D6_RNA_log2TPM', 'CYP2C19_RNA_log2TPM',
                           'ABCB1_RNA_log2TPM', 'IL6_RNA_log2TPM', 'TNF_RNA_log2TPM']
        elif data_type == 'proteomics':
            data = self.get_patient_proteomics(patient_id)
            feature_cols = ['CYP2D6_protein_log2', 'CYP2C19_protein_log2',
                           'ABCB1_protein_log2', 'Albumin_protein_log2', 'CRP_protein_log2']
        else:
            logger.error(f"Unknown data type: {data_type}")
            return None
        
        if data is None or len(data) == 0:
            return None
        
        # Get features that exist in the data
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if len(available_cols) == 0:
            return None
        
        # Extract numeric features
        features = data[available_cols].values.flatten()
        
        # Ensure it's a 2D array (1 sample, n features)
        features = features.reshape(1, -1)
        
        return features
    
    def get_training_data(self, data_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all training data for a specific omics type
        Returns (X, y) where y is drug response
        """
        if self.main_data is None:
            logger.error("No data available")
            return None, None
        
        # Get all patients
        patients = self.get_available_patients()
        
        X_list = []
        y_list = []
        
        for patient_id in patients:
            features = self.get_patient_features(patient_id, data_type)
            if features is not None:
                response = self.get_patient_drug_response(patient_id)
                if response is not None:
                    X_list.append(features.flatten())
                    y_list.append(response)
        
        if len(X_list) == 0:
            logger.error(f"No training data available for {data_type}")
            return None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Training data for {data_type}: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Response range: [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        return X, y
    
    def get_patient_summary(self, patient_id: str) -> Dict:
        """Get summary of available data for a patient"""
        summary = {
            'patient_id': patient_id,
            'exists': self.patient_exists(patient_id),
            'data_available': {
                'genomics': False,
                'transcriptomics': False,
                'proteomics': False,
                'drug_response': False
            },
            'data_shapes': {},
            'clinical_info': {}
        }
        
        if not self.patient_exists(patient_id):
            return summary
        
        # Get patient data
        patient_data = self.main_data[self.main_data['patient_id'] == patient_id].iloc[0]
        
        # Clinical information
        if 'age' in patient_data:
            summary['clinical_info']['age'] = int(patient_data['age'])
        if 'sex' in patient_data:
            summary['clinical_info']['sex'] = patient_data['sex']
        if 'weight_kg' in patient_data:
            summary['clinical_info']['weight'] = float(patient_data['weight_kg'])
        
        # Check genomics
        genomics = self.get_patient_genomics(patient_id)
        if genomics is not None and len(genomics) > 0:
            summary['data_available']['genomics'] = True
            summary['data_shapes']['genomics'] = genomics.shape
        
        # Check transcriptomics
        transcriptomics = self.get_patient_transcriptomics(patient_id)
        if transcriptomics is not None and len(transcriptomics) > 0:
            summary['data_available']['transcriptomics'] = True
            summary['data_shapes']['transcriptomics'] = transcriptomics.shape
        
        # Check proteomics
        proteomics = self.get_patient_proteomics(patient_id)
        if proteomics is not None and len(proteomics) > 0:
            summary['data_available']['proteomics'] = True
            summary['data_shapes']['proteomics'] = proteomics.shape
        
        # Check drug response
        drug_response = self.get_patient_drug_response(patient_id)
        if drug_response is not None:
            summary['data_available']['drug_response'] = True
            summary['actual_drug_response'] = drug_response
            summary['response_class'] = self.get_patient_response_class(patient_id)
        
        return summary


# Global instance
real_data_loader = None

def get_real_data_loader(data_dir: str = ".") -> RealDatasetLoader:
    """Get or create the global data loader instance"""
    global real_data_loader
    if real_data_loader is None:
        real_data_loader = RealDatasetLoader(data_dir)
    return real_data_loader
