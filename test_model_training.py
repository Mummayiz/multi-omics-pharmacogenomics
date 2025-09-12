#!/usr/bin/env python3
"""
Test script to verify model training and prediction functionality
"""

import requests
import json
import time

def test_model_training():
    """Test model training"""
    print("🧪 Testing Model Training")
    print("-" * 40)
    
    url = "http://localhost:8000/api/v1/models/train"
    
    # Test training request
    training_data = {
        "model_type": "multi_omics_fusion",
        "data_types": ["genomics", "transcriptomics", "proteomics"],
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "cross_validation_folds": 3
    }
    
    try:
        response = requests.post(url, json=training_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model training started successfully!")
            print(f"Job ID: {result.get('job_id', 'N/A')}")
            print(f"Response: {result}")
            return result.get('job_id')
        else:
            print(f"❌ Training failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_prediction():
    """Test drug response prediction"""
    print("\n🧪 Testing Drug Response Prediction")
    print("-" * 40)
    
    url = "http://localhost:8000/api/v1/analysis/predict"
    
    # Test prediction request
    prediction_data = {
        "patient_id": "TEST_GEN_15",
        "drug_id": "erlotinib",
        "omics_data_types": ["genomics"],
        "model_type": "multi_omics_fusion"
    }
    
    try:
        response = requests.post(url, json=prediction_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_biomarker_discovery():
    """Test biomarker discovery"""
    print("\n🧪 Testing Biomarker Discovery")
    print("-" * 40)
    
    url = "http://localhost:8000/api/v1/analysis/biomarkers"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Biomarker discovery successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Biomarker discovery failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests"""
    print("🚀 Testing Multi-Omics Platform Model Functionality")
    print("=" * 60)
    
    # Test model training
    job_id = test_model_training()
    
    # Wait a bit for training to start
    if job_id:
        print(f"\n⏳ Waiting for training to complete...")
        time.sleep(5)
    
    # Test prediction
    test_prediction()
    
    # Test biomarker discovery
    test_biomarker_discovery()
    
    print("\n" + "=" * 60)
    print("✅ Model functionality test completed!")

if __name__ == "__main__":
    main()
