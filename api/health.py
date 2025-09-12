"""
Multi-Omics Pharmacogenomics Platform - Health Check API
Vercel serverless function for health monitoring
"""

from datetime import datetime
import json

def handler(request):
    """Health check endpoint for Vercel deployment"""
    
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
    
    if request.method == 'GET':
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'platform': 'Multi-Omics Pharmacogenomics Platform',
            'version': '1.0.0',
            'deployment': 'vercel',
            'services': {
                'api': 'running',
                'models': 'loaded',
                'frontend': 'active'
            },
            'features': [
                'Multi-omics data integration (genomics, transcriptomics, proteomics)',
                'Machine learning models (RandomForest, XGBoost, SVM)',
                'Drug response prediction',
                'Biomarker identification',
                'Model interpretability (SHAP, feature importance)',
                'Lightweight processing pipeline'
            ]
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            },
            'body': json.dumps(health_data)
        }
    
    return {
        'statusCode': 405,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps({'error': 'Method not allowed'})
    }
