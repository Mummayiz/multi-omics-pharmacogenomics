"""
End-to-end system test for Multi-Omics Pharmacogenomics Platform
"""

import os
import sys
import tempfile
import asyncio
import pandas as pd
import numpy as np

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

def test_database():
    """Test database functionality"""
    print("🧪 Testing database...")
    
    try:
        from database.database import db
        
        # Test patient creation
        success = db.create_patient("TEST_001", {"age": 45, "gender": "F"})
        print(f"  ✓ Patient creation: {success}")
        
        # Test patient retrieval
        patient = db.get_patient("TEST_001")
        print(f"  ✓ Patient retrieval: {patient is not None}")
        
        # Test stats
        stats = db.get_database_stats()
        print(f"  ✓ Database stats: {stats['total_patients']} patients")
        
        return True
    except Exception as e:
        print(f"  ✗ Database test failed: {e}")
        return False

def test_data_processing():
    """Test data processing pipeline"""
    print("🧪 Testing data processing...")
    
    try:
        from data_processing.pipeline import processor
        
        # Test sample data generation
        genomics_data = processor.create_sample_data('genomics', n_features=50, n_samples=10)
        print(f"  ✓ Genomics sample data: {genomics_data.shape}")
        
        transcriptomics_data = processor.create_sample_data('transcriptomics', n_features=100, n_samples=10)
        print(f"  ✓ Transcriptomics sample data: {transcriptomics_data.shape}")
        
        proteomics_data = processor.create_sample_data('proteomics', n_features=30, n_samples=10)
        print(f"  ✓ Proteomics sample data: {proteomics_data.shape}")
        
        # Test data processing
        processed_data, metadata = processor.processors['genomics'].process(genomics_data.T)
        print(f"  ✓ Genomics processing: {processed_data.shape} -> {len(metadata['processing_steps'])} steps")
        
        processed_data, metadata = processor.processors['transcriptomics'].process(transcriptomics_data)
        print(f"  ✓ Transcriptomics processing: {processed_data.shape} -> {len(metadata['processing_steps'])} steps")
        
        return True
    except Exception as e:
        print(f"  ✗ Data processing test failed: {e}")
        return False

def test_models():
    """Test machine learning models"""
    print("🧪 Testing ML models...")
    
    try:
        from models.lightweight_models import create_lightweight_model, DEFAULT_CONFIG
        
        # Test genomics model
        model = create_lightweight_model('genomics', DEFAULT_CONFIG.copy())
        print(f"  ✓ Genomics model created: {type(model).__name__}")
        
        # Test transcriptomics model
        model = create_lightweight_model('transcriptomics', DEFAULT_CONFIG.copy())
        print(f"  ✓ Transcriptomics model created: {type(model).__name__}")
        
        # Test proteomics model
        model = create_lightweight_model('proteomics', DEFAULT_CONFIG.copy())
        print(f"  ✓ Proteomics model created: {type(model).__name__}")
        
        # Test fusion model
        model = create_lightweight_model('multi_omics_fusion', DEFAULT_CONFIG.copy())
        print(f"  ✓ Multi-omics fusion model created: {type(model).__name__}")
        
        # Test model training with sample data
        from data_processing.pipeline import processor
        sample_data = processor.create_sample_data('genomics', n_features=50, n_samples=100).T
        y = np.random.randn(100)
        
        genomics_model = create_lightweight_model('genomics', DEFAULT_CONFIG.copy())
        genomics_model.fit(sample_data, y)
        predictions = genomics_model.predict(sample_data[:10])
        print(f"  ✓ Model training and prediction: {len(predictions)} predictions")
        
        return True
    except Exception as e:
        print(f"  ✗ ML models test failed: {e}")
        return False

def test_api_routes():
    """Test API route functionality"""
    print("🧪 Testing API routes...")
    
    try:
        from api.functional_routes import generate_default_prediction, generate_biomarkers
        
        # Test prediction generation
        patient_data = {
            'genomics': pd.DataFrame(np.random.randn(100, 10)),
            'transcriptomics': pd.DataFrame(np.random.randn(200, 10))
        }
        
        prediction = generate_default_prediction(patient_data, 'erlotinib')
        print(f"  ✓ Default prediction: {prediction['predicted_response']:.3f} confidence: {prediction['confidence_score']:.3f}")
        
        # Test biomarker generation
        biomarkers = generate_biomarkers(patient_data, 'erlotinib')
        print(f"  ✓ Biomarker generation: {len(biomarkers)} biomarkers")
        
        return True
    except Exception as e:
        print(f"  ✗ API routes test failed: {e}")
        return False

def test_file_operations():
    """Test file upload and processing simulation"""
    print("🧪 Testing file operations...")
    
    try:
        from data_processing.pipeline import processor
        from database.database import db
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            # Create sample genomics data
            sample_data = processor.create_sample_data('genomics', n_features=20, n_samples=5)
            sample_data.to_csv(tmp.name)
            temp_file = tmp.name
        
        try:
            # Test file processing
            processed_data, metadata = processor.process_file(temp_file, 'genomics', 'TEST_002')
            print(f"  ✓ File processing: {processed_data.shape} -> {metadata['processing_type']}")
            
            # Test database storage
            processed_path = db.store_processed_data('TEST_002', 'genomics', processed_data)
            print(f"  ✓ Database storage: {os.path.basename(processed_path)}")
            
            # Test data retrieval
            retrieved_data = db.load_processed_data('TEST_002', 'genomics')
            print(f"  ✓ Data retrieval: {retrieved_data.shape if retrieved_data is not None else 'None'}")
            
        finally:
            # Clean up
            os.unlink(temp_file)
        
        return True
    except Exception as e:
        print(f"  ✗ File operations test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Multi-Omics Pharmacogenomics Platform - System Test")
    print("=" * 60)
    
    tests = [
        ("Database", test_database),
        ("Data Processing", test_data_processing),
        ("Machine Learning Models", test_models),
        ("API Routes", test_api_routes),
        ("File Operations", test_file_operations)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}")
        result = test_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! The system is functional.")
        print("\n📖 Next steps:")
        print("1. Start the backend: cd backend && python main.py")
        print("2. Open frontend/index.html in your browser")
        print("3. Test the web interface:")
        print("   - Upload sample data files")
        print("   - Train models")
        print("   - Make predictions")
        print("   - View visualizations")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
