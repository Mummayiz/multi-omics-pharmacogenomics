"""
Test script to verify all functionality before deployment
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_patient_prediction(patient_id):
    """Test prediction for a specific patient"""
    print(f"\n=== Testing Prediction for {patient_id} ===")
    try:
        payload = {
            "patient_id": patient_id,
            "data_type": "genomics",
            "drug_name": "DrugX"
        }
        response = requests.post(
            f"{BASE_URL}/predict/drug-response",
            json=payload,
            timeout=10
        )
        print(f"✓ Status: {response.status_code}")
        result = response.json()
        print(f"✓ Predicted Score: {result.get('predicted_score', 'N/A')}")
        print(f"✓ Response Class: {result.get('response_class', 'N/A')}")
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_invalid_patient():
    """Test error handling for invalid patient ID"""
    print("\n=== Testing Invalid Patient ID ===")
    try:
        payload = {
            "patient_id": "INVALID_999",
            "data_type": "genomics",
            "drug_name": "DrugX"
        }
        response = requests.post(
            f"{BASE_URL}/predict/drug-response",
            json=payload,
            timeout=10
        )
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Error Response: {response.json()}")
        return response.status_code == 404
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_patient_specific_predictions():
    """Test that different patients get different predictions"""
    print("\n=== Testing Patient-Specific Predictions ===")
    patients = ["P001", "P002", "P025", "P050"]
    predictions = {}
    
    for patient_id in patients:
        result = test_patient_prediction(patient_id)
        if result:
            predictions[patient_id] = result.get('predicted_score')
    
    # Check if predictions are different
    unique_predictions = len(set(predictions.values()))
    print(f"\n✓ Tested {len(patients)} patients")
    print(f"✓ Got {unique_predictions} unique predictions")
    
    if unique_predictions > 1:
        print("✓ SUCCESS: Patients have different predictions!")
        return True
    else:
        print("✗ FAIL: All patients have same prediction")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("MULTI-OMICS PLATFORM - PRE-DEPLOYMENT TESTS")
    print("="*60)
    
    # Check if server is running
    print("\nWaiting for server to start...")
    for i in range(5):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            print("✓ Server is ready!")
            break
        except:
            print(f"  Waiting... ({i+1}/5)")
            time.sleep(2)
    else:
        print("\n✗ Server is not running!")
        print("Please start the server with: cd backend && python main.py")
        return
    
    # Run tests
    results = {
        "health": test_health(),
        "patient_specific": test_patient_specific_predictions(),
        "error_handling": test_invalid_patient()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Ready for deployment!")
    else:
        print("\n⚠️ SOME TESTS FAILED! Please fix before deploying.")
    
    return all_passed

if __name__ == "__main__":
    main()
