"""
Configuration settings for Multi-Omics Pharmacogenomics Platform
"""

import os
from typing import Dict, Any

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Multi-Omics Pharmacogenomics Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API
    API_PREFIX: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./multi_omics.db")
    
    # ML Models
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/saved")
    MAX_MODEL_SIZE: int = int(os.getenv("MAX_MODEL_SIZE", "1000000000"))  # 1GB
    
    # Data Processing
    DATA_PATH: str = os.getenv("DATA_PATH", "./data")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "500000000"))  # 500MB
    SUPPORTED_FORMATS: list = ["vcf", "bam", "csv", "tsv", "h5", "hdf5"]
    
    # Deep Learning
    DEVICE: str = os.getenv("DEVICE", "cpu")  # cpu, cuda, mps
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_EPOCHS: int = int(os.getenv("MAX_EPOCHS", "100"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    
    # Genomics Processing
    REFERENCE_GENOME: str = os.getenv("REFERENCE_GENOME", "GRCh38")
    MIN_QUALITY_SCORE: int = int(os.getenv("MIN_QUALITY_SCORE", "30"))
    
    # Cloud Storage (if using)
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Dataset configurations
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "genomics": {
        "1000genomes": {
            "url": "https://www.internationalgenome.org/data",
            "format": "vcf",
            "description": "Human genetic variation data"
        },
        "tcga": {
            "url": "https://portal.gdc.cancer.gov/",
            "format": "vcf",
            "description": "Cancer genomics data"
        }
    },
    "transcriptomics": {
        "tcga_rnaseq": {
            "url": "https://portal.gdc.cancer.gov/",
            "format": "csv",
            "description": "RNA-seq expression data"
        },
        "gtex": {
            "url": "https://gtexportal.org/",
            "format": "csv",
            "description": "Tissue-specific expression"
        }
    },
    "proteomics": {
        "human_protein_atlas": {
            "url": "https://www.proteinatlas.org/",
            "format": "csv",
            "description": "Protein abundance data"
        }
    },
    "drug_response": {
        "gdsc": {
            "url": "https://www.cancerrxgene.org/",
            "format": "csv",
            "description": "Drug sensitivity data"
        },
        "pharmgkb": {
            "url": "https://www.pharmgkb.org/",
            "format": "csv",
            "description": "Pharmacogenomics data"
        }
    }
}

# Model architecture configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "genomics_cnn": {
        "input_shape": (23, 1000000, 1),  # (chromosomes, positions, channels)
        "conv_layers": [64, 128, 256],
        "kernel_sizes": [3, 5, 7],
        "dropout_rate": 0.3
    },
    "transcriptomics_rnn": {
        "input_shape": (None, 20000),  # (time_points, genes)
        "rnn_units": [128, 64],
        "rnn_type": "LSTM",
        "dropout_rate": 0.4
    },
    "proteomics_fc": {
        "input_shape": (10000,),  # number of proteins
        "hidden_layers": [512, 256, 128],
        "activation": "relu",
        "dropout_rate": 0.3
    },
    "multi_omics_fusion": {
        "attention_heads": 8,
        "attention_dim": 64,
        "fusion_type": "late",  # early, late, intermediate
        "final_layers": [256, 128, 64]
    }
}
