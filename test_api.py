#!/usr/bin/env python3
"""
Test the Flask API endpoint directly
"""

import requests
import os
from pathlib import Path

# Get test image path
upload_folder = Path('web/static/uploads')
test_image = next(upload_folder.glob('*.jpg'), None)

if not test_image:
    print("❌ No test image found!")
    exit(1)

print(f"📷 Testing API with: {test_image.name}")

# Upload to /predict endpoint
url = 'http://localhost:5000/predict'
with open(test_image, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print(f"\n📡 Response Status: {response.status_code}")
print(f"📡 Response Headers: {dict(response.headers)}")
print(f"\n📊 Response JSON:")

import json
result = response.json()
print(json.dumps(result, indent=2))

# Verify response structure
print("\n✅ Response verification:")
print(f"   - success: {result.get('success')}")
print(f"   - class: {result.get('class')}")
print(f"   - predicted_class: {result.get('predicted_class')}")
print(f"   - confidence: {result.get('confidence')} (type: {type(result.get('confidence')).__name__})")
print(f"   - is_defect: {result.get('is_defect')}")
print(f"   - probabilities keys: {list(result.get('probabilities', {}).keys())}")
print(f"   - probabilities sample: {dict(list(result.get('probabilities', {}).items())[:3])}")
