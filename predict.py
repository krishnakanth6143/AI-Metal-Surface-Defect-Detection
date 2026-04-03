"""
Prediction and inference module for metal surface defects.
Handles model loading and making predictions on new images.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.preprocessing import ImageProcessor


class DefectPredictor:
    """Handles prediction and inference."""
    
    def __init__(self, model_path=CNN_MODEL_PATH):
        """
        Initialize predictor with pre-trained model.
        
        Args:
            model_path (str): Path to saved model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = list(DEFECT_CLASSES.values())
        self.processor = ImageProcessor(target_size=(IMG_SIZE, IMG_SIZE))
        self.confidence_threshold = 0.5
        
        self.load_model()
    
    def load_model(self):
        """Load model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Model loaded from {self.model_path}")
    
    def preprocess_for_prediction(self, image_path):
        """
        Preprocess image for model prediction.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (preprocessed_image, original_image_for_display)
        """
        preprocessed = self.processor.preprocess_image(image_path, normalize=True)
        original, edges, resized = self.processor.preprocess_for_analysis(image_path)
        
        # Add batch dimension
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return preprocessed, original, resized, edges
    
    def predict(self, image_path):
        """
        Predict defect class for an image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        image_batch, original, resized, edges = self.preprocess_for_prediction(image_path)
        
        # Predict
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all probabilities
        probabilities = {self.class_names[i]: float(predictions[0][i]) 
                        for i in range(len(self.class_names))}
        
        # Determine if defect is detected
        is_defect = self.class_names[predicted_class_idx] != 'Normal'
        
        result = {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': confidence,
            'is_defect': is_defect,
            'all_probabilities': probabilities,
            'class_index': int(predicted_class_idx),
            'confidence_score': f"{confidence * 100:.2f}%"
        }
        
        return result
    
    def batch_predict(self, image_dir):
        """
        Predict on multiple images.
        
        Args:
            image_dir (str): Directory containing images
            
        Returns:
            list: List of prediction results
        """
        results = []
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        print(f"🔍 Predicting on {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                result = self.predict(img_path)
                result['filename'] = img_file
                results.append(result)
            except Exception as e:
                print(f"❌ Error predicting on {img_file}: {str(e)}")
        
        return results
    
    def get_defect_summary(self, results):
        """
        Get summary of predictions.
        
        Args:
            results (list): List of prediction results
            
        Returns:
            dict: Summary statistics
        """
        total = len(results)
        defects = sum(1 for r in results if r['is_defect'])
        normal = total - defects
        
        class_counts = {}
        for r in results:
            class_name = r['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_images': total,
            'defects_found': defects,
            'normal_surfaces': normal,
            'defect_percentage': (defects / total * 100) if total > 0 else 0,
            'class_distribution': class_counts
        }


def predict_single_image(image_path, model_path=CNN_MODEL_PATH):
    """
    Convenience function to predict on a single image.
    
    Args:
        image_path (str): Path to image
        model_path (str): Path to model
        
    Returns:
        dict: Prediction results
    """
    predictor = DefectPredictor(model_path)
    return predictor.predict(image_path)


if __name__ == '__main__':
    # Example usage
    print("Metal Surface Defect Detection - Inference")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"❌ Model not found at {CNN_MODEL_PATH}")
        print("Please train the model first using train.py")
    else:
        predictor = DefectPredictor()
        print("✅ Predictor initialized successfully!")
