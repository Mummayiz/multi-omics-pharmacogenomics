"""
Multi-Omics Pharmacogenomics Platform - Drug Response Prediction API
Vercel serverless function for making predictions
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import base64
import io
import sys
import os

# Add the models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'models'))

try:
    from lightweight_models import create_lightweight_model, DEFAULT_CONFIG
except ImportError:
    # Fallback for simple prediction without models
    def create_lightweight_model(model_type, config):
        return None
    DEFAULT_CONFIG = {}

def generate_mock_prediction(patient_data):
    """Generate a mock prediction for demonstration purposes"""
    # Simple mock prediction based on input features
    np.random.seed(42)  # For reproducible results
    
    # Extract feature counts
    genomics_features = len(patient_data.get('genomics', {}))
    transcriptomics_features = len(patient_data.get('transcriptomics', {}))
    proteomics_features = len(patient_data.get('proteomics', {}))
    
    # Generate mock prediction
    base_response = 0.5 + (genomics_features * 0.001) + (transcriptomics_features * 0.0001) + (proteomics_features * 0.01)
    base_response = min(max(base_response, 0.0), 1.0)  # Clamp between 0 and 1
    
    # Add some realistic noise
    noise = np.random.normal(0, 0.05)
    predicted_response = min(max(base_response + noise, 0.0), 1.0)
    
    # Generate confidence score
    confidence = 0.7 + np.random.random() * 0.25  # Between 0.7 and 0.95
    
    # Generate feature importance
    feature_importance = {
        'genomics': {
            'EGFR': 0.25,
            'KRAS': 0.18,
            'TP53': 0.15,
            'BRCA1': 0.12,
            'MYC': 0.10
        },
        'transcriptomics': {
            'gene_expression_signature_1': 0.08,
            'gene_expression_signature_2': 0.06,
            'pathway_activation_score': 0.04
        },
        'proteomics': {
            'protein_abundance_marker_1': 0.03,
            'protein_abundance_marker_2': 0.02
        }
    }
    
    # Generate biomarkers
    biomarkers = [
        {
            'name': 'EGFR mutation status',
            'type': 'genomic',
            'importance': 0.25,
            'description': 'Key predictor for EGFR inhibitor response'
        },
        {
            'name': 'Gene expression signature',
            'type': 'transcriptomic',
            'importance': 0.08,
            'description': 'Multi-gene expression pattern associated with drug response'
        },
        {
            'name': 'Protein marker',
            'type': 'proteomic',
            'importance': 0.03,
            'description': 'Protein abundance level correlating with drug efficacy'
        }
    ]
    
    return {
        'predicted_response': round(predicted_response, 4),
        'confidence_score': round(confidence, 4),
        'feature_importance': feature_importance,
        'biomarkers': biomarkers
    }

def handler(request):
    """Drug response prediction endpoint"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            },
            'body': ''
        }
    
    if request.method == 'POST':
        try:
            # Parse request body
            if hasattr(request, 'body'):
                if isinstance(request.body, bytes):
                    body = json.loads(request.body.decode('utf-8'))
                else:
                    body = json.loads(request.body)
            else:
                body = request.json if hasattr(request, 'json') else {}
            
            # Validate required fields
            patient_id = body.get('patient_id')
            drug_id = body.get('drug_id')
            omics_data = body.get('omics_data', {})
            
            if not patient_id or not drug_id:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                    },
                    'body': json.dumps({
                        'error': 'Missing required fields: patient_id and drug_id'
                    })
                }
            
            # Generate prediction (using mock data for demo)
            prediction_result = generate_mock_prediction(omics_data)
            
            # Create response
            response_data = {
                'patient_id': patient_id,
                'drug_id': drug_id,
                'prediction_id': f"PRED_{patient_id}_{drug_id}_{int(datetime.now().timestamp())}",
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction_result,
                'model_info': {
                    'model_type': 'multi_omics_fusion',
                    'model_version': '1.0.0',
                    'features_used': list(omics_data.keys()),
                    'deployment': 'vercel'
                }
            }
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                },
                'body': json.dumps(response_data)
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
                'body': json.dumps({
                    'error': 'Internal server error',
                    'message': str(e)
                })
            }
    
    elif request.method == 'GET':
        # Return API information
        api_info = {
            'endpoint': '/api/predict',
            'method': 'POST',
            'description': 'Predict drug response using multi-omics data',
            'required_fields': ['patient_id', 'drug_id'],
            'optional_fields': ['omics_data'],
            'example_request': {
                'patient_id': 'PATIENT_001',
                'drug_id': 'erlotinib',
                'omics_data': {
                    'genomics': {
                        'EGFR': 'mutant',
                        'KRAS': 'wild_type',
                        'TP53': 'mutant'
                    },
                    'transcriptomics': {
                        'gene_expression_signature_1': 0.75,
                        'gene_expression_signature_2': 0.45
                    },
                    'proteomics': {
                        'protein_abundance_marker_1': 1.2,
                        'protein_abundance_marker_2': 0.8
                    }
                }
            },
            'response_format': {
                'patient_id': 'string',
                'drug_id': 'string',
                'prediction_id': 'string',
                'timestamp': 'ISO datetime',
                'prediction': {
                    'predicted_response': 'float (0-1)',
                    'confidence_score': 'float (0-1)',
                    'feature_importance': 'object',
                    'biomarkers': 'array'
                }
            }
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps(api_info, indent=2)
        }
    
    return {
        'statusCode': 405,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps({'error': 'Method not allowed'})
    }
