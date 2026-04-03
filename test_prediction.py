#!/usr/bin/env python3
"""
Quick test script to verify predictions are working correctly.
"""

import os
import sys
import json
from pathlib import Path

# Set up path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from predict import DefectPredictor

# Find test images from uploaded folder
upload_folder = os.path.join(os.path.dirname(__file__), 'web', 'static', 'uploads')
test_images = []

for img_file in os.listdir(upload_folder):
    if img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        test_images.append(os.path.join(upload_folder, img_file))

if not test_images:
    print("❌ No test images found in uploads folder!")
    sys.exit(1)

print(f"✅ Found {len(test_images)} test images")

# Initialize predictor
print("\n[INFO] Loading model...")
predictor = DefectPredictor(CNN_MODEL_PATH)
print("[SUCCESS] Model loaded!\n")

# Test predictions
print("=" * 70)
print("TESTING PREDICTIONS")
print("=" * 70)

for image_path in test_images[:5]:  # Test first 5 images
    filename = os.path.basename(image_path)
    print(f"\n📷 Testing: {filename}")
    
    try:
        result = predictor.predict(image_path)
        print(f"   ✅ Prediction: {result['predicted_class']}")
        print(f"   📊 Confidence: {result['confidence_score']} ({result['confidence']:.4f})")
        print(f"   🎯 Is Defect: {result['is_defect']}")
        print(f"   📈 All Probabilities:")
        
        # Pretty print probabilities
        for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar_len = int(prob * 30)
            bar = '█' * bar_len
            print(f"      {cls:20s}: {prob:.4f} {bar}")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
