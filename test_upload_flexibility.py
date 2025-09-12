#!/usr/bin/env python3
"""
Test script to verify upload functionality with different column counts
"""

import requests
import os
import time

def test_upload(file_path, patient_id, data_type):
    """Test upload with a specific file"""
    print(f"\n🧪 Testing upload: {file_path}")
    print(f"   Patient ID: {patient_id}")
    print(f"   Data Type: {data_type}")
    
    url = "http://localhost:8000/api/v1/omics/upload"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            params = {
                'patient_id': patient_id,
                'data_type': data_type
            }
            
            response = requests.post(url, files=files, params=params)
            
            print(f"   Status Code: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ Upload successful!")
                result = response.json()
                print(f"   Response: {result}")
            else:
                print(f"   ❌ Upload failed: {response.text}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Test all sample files"""
    print("🚀 Testing Multi-Omics Platform Upload Flexibility")
    print("=" * 60)
    
    # Test files with different column counts
    test_cases = [
        ("sample_genomics_data.csv", "TEST_GEN_15", "genomics"),  # 15 columns
        ("sample_genomics_20cols.csv", "TEST_GEN_20", "genomics"),  # 20 columns
        ("sample_transcriptomics_5cols.csv", "TEST_TRANS_5", "transcriptomics"),  # 5 columns
        ("sample_proteomics_8cols.csv", "TEST_PROT_8", "proteomics"),  # 8 columns
        ("sample_drug_response_3cols.csv", "TEST_DRUG_3", "drug_response"),  # 3 columns
    ]
    
    for file_path, patient_id, data_type in test_cases:
        if os.path.exists(file_path):
            test_upload(file_path, patient_id, data_type)
            time.sleep(1)  # Wait between uploads
        else:
            print(f"❌ File not found: {file_path}")
    
    print("\n" + "=" * 60)
    print("✅ Upload flexibility test completed!")

if __name__ == "__main__":
    main()
