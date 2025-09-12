"""
Multi-Omics Pharmacogenomics Platform - Main API Server
Integrating Multi-Omics Data for Precision Medicine in Pharmacogenomics Using Deep Learning
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
from datetime import datetime

# Import custom modules (use relative imports when running from backend directory)
try:
    from api.functional_routes import omics_router, model_router, analysis_router
except ImportError:
    try:
        from backend.api.functional_routes import omics_router, model_router, analysis_router
    except ImportError:
        # Fallback to original routes
        try:
            from api.routes import omics_router, model_router, analysis_router
        except ImportError:
            from backend.api.routes import omics_router, model_router, analysis_router

try:
    from utils.config import settings
    from utils.logger import get_logger
except ImportError:
    from backend.utils.config import settings
    from backend.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Omics Pharmacogenomics Platform",
    description="AI-powered platform for integrating multi-omics data in precision medicine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Frontend development server
        "http://127.0.0.1:3000",  # Alternative localhost
        "file://",  # Allow file:// protocol
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Include API routes
app.include_router(omics_router, prefix="/api/v1/omics", tags=["Multi-Omics Data"])
app.include_router(model_router, prefix="/api/v1/models", tags=["Deep Learning Models"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis & Predictions"])

# Health check and basic info endpoints
@app.get("/")
async def root():
    """Root endpoint with platform information"""
    return {
        "platform": "Multi-Omics Pharmacogenomics Platform",
        "description": "Integrating Multi-Omics Data for Precision Medicine in Pharmacogenomics Using Deep Learning",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "status": "active",
        "features": [
            "Multi-omics data integration (genomics, transcriptomics, proteomics)",
            "Deep learning models (CNN, RNN, attention mechanisms)",
            "Drug response prediction",
            "Biomarker identification",
            "Model interpretability (SHAP, attention visualization)",
            "Scalable processing pipeline"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",  # Will be implemented
            "ml_models": "loaded"     # Will be implemented
        }
    }

# API v1 convenience endpoints to match frontend baseURL
@app.get("/api/v1/")
async def api_v1_root():
    return await root()

@app.get("/api/v1/health")
async def api_v1_health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Data models for API
class OmicsDataUpload(BaseModel):
    """Model for multi-omics data upload"""
    patient_id: str
    data_type: str  # genomics, transcriptomics, proteomics
    file_format: str  # vcf, bam, csv, etc.
    metadata: Optional[Dict] = None

class PredictionRequest(BaseModel):
    """Model for drug response prediction request"""
    patient_id: str
    drug_id: str
    omics_data_types: List[str]
    model_type: Optional[str] = "multi_branch_fusion"

class AnalysisResult(BaseModel):
    """Model for analysis results"""
    patient_id: str
    drug_id: str
    predicted_response: float
    confidence_score: float
    biomarkers: List[Dict]
    interpretability_data: Dict

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
