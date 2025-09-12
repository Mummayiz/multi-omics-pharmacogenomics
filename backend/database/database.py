"""
Multi-Omics Pharmacogenomics Platform - Database Layer
Simple SQLite-based database for lightweight deployment
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import pandas as pd
import numpy as np

class MultiOmicsDatabase:
    """Lightweight SQLite database for multi-omics data management"""
    
    def __init__(self, db_path: str = "multi_omics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Omics data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS omics_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    data_type TEXT,
                    file_name TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    processing_status TEXT DEFAULT 'uploaded',
                    processed_data_path TEXT,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT,
                    model_config TEXT,
                    training_status TEXT DEFAULT 'initialized',
                    model_path TEXT,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    drug_id TEXT,
                    model_id TEXT,
                    prediction_result TEXT,
                    confidence_score REAL,
                    biomarkers TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Patient management
    def create_patient(self, patient_id: str, metadata: Dict = None) -> bool:
        """Create a new patient record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO patients (patient_id, metadata)
                    VALUES (?, ?)
                """, (patient_id, json.dumps(metadata or {})))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Patient already exists
                return False
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get patient information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'patient_id': row['patient_id'],
                    'created_at': row['created_at'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'updated_at': row['updated_at']
                }
            return None
    
    def list_patients(self) -> List[Dict]:
        """List all patients"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    # Omics data management
    def add_omics_data(self, patient_id: str, data_type: str, file_name: str, 
                      file_path: str, file_size: int, metadata: Dict = None) -> int:
        """Add omics data record"""
        # Ensure patient exists
        if not self.get_patient(patient_id):
            self.create_patient(patient_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO omics_data 
                (patient_id, data_type, file_name, file_path, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (patient_id, data_type, file_name, file_path, file_size, 
                 json.dumps(metadata or {})))
            conn.commit()
            return cursor.lastrowid
    
    def get_patient_omics_data(self, patient_id: str) -> List[Dict]:
        """Get all omics data for a patient"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM omics_data 
                WHERE patient_id = ? 
                ORDER BY upload_timestamp DESC
            """, (patient_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def update_omics_processing_status(self, data_id: int, status: str, 
                                     processed_data_path: str = None):
        """Update processing status of omics data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE omics_data 
                SET processing_status = ?, processed_data_path = ?
                WHERE id = ?
            """, (status, processed_data_path, data_id))
            conn.commit()
    
    # Model management
    def create_model(self, model_id: str, model_type: str, model_config: Dict) -> bool:
        """Create a new model record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO models (model_id, model_type, model_config)
                    VALUES (?, ?, ?)
                """, (model_id, model_type, json.dumps(model_config)))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def update_model_status(self, model_id: str, status: str, 
                           model_path: str = None, 
                           performance_metrics: Dict = None):
        """Update model training status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            completed_at = datetime.now().isoformat() if status == 'completed' else None
            cursor.execute("""
                UPDATE models 
                SET training_status = ?, model_path = ?, 
                    performance_metrics = ?, completed_at = ?
                WHERE model_id = ?
            """, (status, model_path, 
                 json.dumps(performance_metrics) if performance_metrics else None,
                 completed_at, model_id))
            conn.commit()
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'model_id': row['model_id'],
                    'model_type': row['model_type'],
                    'model_config': json.loads(row['model_config']) if row['model_config'] else {},
                    'training_status': row['training_status'],
                    'model_path': row['model_path'],
                    'performance_metrics': json.loads(row['performance_metrics']) if row['performance_metrics'] else {},
                    'created_at': row['created_at'],
                    'completed_at': row['completed_at']
                }
            return None
    
    def list_models(self) -> List[Dict]:
        """List all models"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    # Prediction management
    def save_prediction(self, prediction_id: str, patient_id: str, drug_id: str,
                       model_id: str, prediction_result: Dict, 
                       confidence_score: float, biomarkers: List[Dict]) -> bool:
        """Save prediction results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions 
                (prediction_id, patient_id, drug_id, model_id, 
                 prediction_result, confidence_score, biomarkers)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (prediction_id, patient_id, drug_id, model_id,
                 json.dumps(prediction_result), confidence_score,
                 json.dumps(biomarkers)))
            conn.commit()
            return True
    
    def get_prediction(self, prediction_id: str) -> Optional[Dict]:
        """Get prediction results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE prediction_id = ?", 
                          (prediction_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'prediction_id': row['prediction_id'],
                    'patient_id': row['patient_id'],
                    'drug_id': row['drug_id'],
                    'model_id': row['model_id'],
                    'prediction_result': json.loads(row['prediction_result']),
                    'confidence_score': row['confidence_score'],
                    'biomarkers': json.loads(row['biomarkers']),
                    'created_at': row['created_at']
                }
            return None
    
    def get_patient_predictions(self, patient_id: str) -> List[Dict]:
        """Get all predictions for a patient"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE patient_id = ? 
                ORDER BY created_at DESC
            """, (patient_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # Data processing helpers
    def store_processed_data(self, patient_id: str, data_type: str, 
                           processed_data: pd.DataFrame, 
                           processing_metadata: Dict = None) -> str:
        """Store processed data as a file and return path"""
        # Create processed data directory if it doesn't exist
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{patient_id}_{data_type}_{timestamp}.csv"
        filepath = os.path.join(processed_dir, filename)
        
        # Save data
        processed_data.to_csv(filepath, index=False)
        
        # Update database record
        omics_records = self.get_patient_omics_data(patient_id)
        for record in omics_records:
            if record['data_type'] == data_type and record['processing_status'] == 'uploaded':
                self.update_omics_processing_status(
                    record['id'], 'processed', filepath
                )
                break
        
        return filepath
    
    def load_processed_data(self, patient_id: str, data_type: str) -> Optional[pd.DataFrame]:
        """Load processed data for a patient and data type"""
        omics_records = self.get_patient_omics_data(patient_id)
        for record in omics_records:
            if (record['data_type'] == data_type and 
                record['processing_status'] == 'processed' and 
                record['processed_data_path']):
                try:
                    return pd.read_csv(record['processed_data_path'])
                except Exception:
                    continue
        return None
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count tables
            cursor.execute("SELECT COUNT(*) FROM patients")
            stats['total_patients'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM omics_data")
            stats['total_omics_files'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM models")
            stats['total_models'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM predictions")
            stats['total_predictions'] = cursor.fetchone()[0]
            
            # Data type breakdown
            cursor.execute("""
                SELECT data_type, COUNT(*) 
                FROM omics_data 
                GROUP BY data_type
            """)
            stats['omics_by_type'] = dict(cursor.fetchall())
            
            # Processing status breakdown
            cursor.execute("""
                SELECT processing_status, COUNT(*) 
                FROM omics_data 
                GROUP BY processing_status
            """)
            stats['processing_status'] = dict(cursor.fetchall())
            
            return stats

# Global database instance
db = MultiOmicsDatabase()
