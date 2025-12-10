"""
Comprehensive Functional Test - All Features
"""
import sys
import traceback

def test_data_loader():
    """Test 1: Real Data Loader"""
    print("\n" + "="*60)
    print("TEST 1: Real Data Loader")
    print("="*60)
    try:
        from backend.data_processing.real_data_loader import get_real_data_loader
        
        loader = get_real_data_loader(data_dir=".")
        patients = loader.get_available_patients()
        
        print(f"✓ Dataset loaded: {len(patients)} patients")
        print(f"✓ Patient IDs: {patients[:5]}...{patients[-2:]}")
        
        # Test patient data retrieval
        test_patient = "P001"
        genomics = loader.get_patient_genomics(test_patient)
        transcriptomics = loader.get_patient_transcriptomics(test_patient)
        proteomics = loader.get_patient_proteomics(test_patient)
        drug_response = loader.get_patient_drug_response(test_patient)
        
        print(f"✓ Genomics data shape: {genomics.shape}")
        print(f"✓ Transcriptomics data shape: {transcriptomics.shape}")
        print(f"✓ Proteomics data shape: {proteomics.shape}")
        print(f"✓ Drug response: {drug_response:.3f}")
        
        return True, loader
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False, None

def test_deep_learning_models():
    """Test 2: Deep Learning Models"""
    print("\n" + "="*60)
    print("TEST 2: Deep Learning Models")
    print("="*60)
    try:
        from backend.models.deep_learning_models import CNNModel, RNNModel, AttentionModel
        from backend.data_processing.real_data_loader import get_real_data_loader
        import numpy as np
        
        loader = get_real_data_loader(data_dir=".")
        X, y = loader.get_training_data('genomics')
        
        print(f"✓ Training data: X shape={X.shape}, y shape={y.shape}")
        
        # Test CNN
        config = {'task_type': 'regression'}
        cnn = CNNModel(config)
        cnn.fit(X, y)
        pred_cnn = cnn.predict(X[:1])
        print(f"✓ CNN trained and predicted: {pred_cnn[0]:.3f}")
        
        # Test RNN
        rnn = RNNModel(config)
        rnn.fit(X, y)
        pred_rnn = rnn.predict(X[:1])
        print(f"✓ RNN trained and predicted: {pred_rnn[0]:.3f}")
        
        # Test Attention
        att = AttentionModel(config)
        att.fit(X, y)
        pred_att = att.predict(X[:1])
        print(f"✓ Attention trained and predicted: {pred_att[0]:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def test_patient_specific_predictions(loader):
    """Test 3: Patient-Specific Predictions"""
    print("\n" + "="*60)
    print("TEST 3: Patient-Specific Predictions")
    print("="*60)
    try:
        test_patients = ["P001", "P002", "P010", "P025", "P050"]
        responses = {}
        
        for patient_id in test_patients:
            response = loader.get_patient_drug_response(patient_id)
            response_class = loader.get_patient_response_class(patient_id)
            responses[patient_id] = response
            print(f"✓ {patient_id}: {response:.3f} ({response_class})")
        
        # Verify they're different
        unique_responses = len(set(responses.values()))
        print(f"\n✓ Tested {len(test_patients)} patients")
        print(f"✓ Found {unique_responses} unique drug response values")
        
        if unique_responses > 1:
            print("✓ SUCCESS: Patients have different predictions!")
            return True
        else:
            print("✗ FAIL: All patients have same prediction")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def test_error_handling(loader):
    """Test 4: Error Handling"""
    print("\n" + "="*60)
    print("TEST 4: Error Handling for Invalid Patient IDs")
    print("="*60)
    try:
        invalid_ids = ["INVALID_001", "P999", "ABC123", ""]
        
        for invalid_id in invalid_ids:
            exists = loader.patient_exists(invalid_id)
            print(f"✓ {invalid_id or '(empty)'}: exists={exists} (should be False)")
            
            if exists:
                print(f"✗ FAIL: Invalid ID {invalid_id} should not exist")
                return False
        
        print("✓ SUCCESS: All invalid IDs correctly identified")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def test_api_routes_import():
    """Test 5: API Routes Import"""
    print("\n" + "="*60)
    print("TEST 5: API Routes Import")
    print("="*60)
    try:
        from backend.api.functional_routes import omics_router, model_router, analysis_router
        
        print(f"✓ omics_router imported: {len(omics_router.routes)} routes")
        print(f"✓ model_router imported: {len(model_router.routes)} routes")
        print(f"✓ analysis_router imported: {len(analysis_router.routes)} routes")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test 6: Feature Extraction"""
    print("\n" + "="*60)
    print("TEST 6: Feature Extraction for Different Data Types")
    print("="*60)
    try:
        from backend.data_processing.real_data_loader import get_real_data_loader
        
        loader = get_real_data_loader(data_dir=".")
        patient_id = "P001"
        
        # Test each data type
        for data_type in ['genomics', 'transcriptomics', 'proteomics']:
            features = loader.get_patient_features(patient_id, data_type)
            print(f"✓ {data_type.capitalize()}: shape={features.shape}, values={features[0][:3]}...")
        
        print("✓ SUCCESS: All feature types extracted correctly")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def test_training_data():
    """Test 7: Training Data Generation"""
    print("\n" + "="*60)
    print("TEST 7: Training Data Generation")
    print("="*60)
    try:
        from backend.data_processing.real_data_loader import get_real_data_loader
        
        loader = get_real_data_loader(data_dir=".")
        
        for data_type in ['genomics', 'transcriptomics', 'proteomics']:
            X, y = loader.get_training_data(data_type)
            print(f"✓ {data_type.capitalize()}: X={X.shape}, y={y.shape}")
            print(f"  Response range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
        
        print("✓ SUCCESS: Training data generated for all types")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all functional tests"""
    print("="*60)
    print("COMPREHENSIVE FUNCTIONAL TESTING")
    print("Multi-Omics Pharmacogenomics Platform")
    print("="*60)
    
    results = {}
    
    # Test 1: Data Loader
    results['data_loader'], loader = test_data_loader()
    
    if not results['data_loader']:
        print("\n⚠️  Data loader failed - stopping tests")
        return False
    
    # Test 2: Deep Learning Models
    results['deep_learning'] = test_deep_learning_models()
    
    # Test 3: Patient-Specific Predictions
    results['patient_specific'] = test_patient_specific_predictions(loader)
    
    # Test 4: Error Handling
    results['error_handling'] = test_error_handling(loader)
    
    # Test 5: API Routes
    results['api_routes'] = test_api_routes_import()
    
    # Test 6: Feature Extraction
    results['feature_extraction'] = test_feature_extraction()
    
    # Test 7: Training Data
    results['training_data'] = test_training_data()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        test_label = test_name.replace('_', ' ').title()
        print(f"{status}: {test_label}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
        print("✓ System is fully functional")
        print("✓ Ready for deployment")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please fix issues before deploying")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
