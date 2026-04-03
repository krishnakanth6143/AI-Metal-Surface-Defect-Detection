"""
Testing and validation utilities.
Provides functions for model testing and performance evaluation.
"""

import os
import sys
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CNN_MODEL_PATH
from predict import DefectPredictor


class ModelTester:
    """Test and validate model performance."""
    
    def __init__(self, model_path=CNN_MODEL_PATH):
        """Initialize tester."""
        self.predictor = DefectPredictor(model_path)
        self.results = []
    
    def test_single_image(self, image_path):
        """
        Test on single image.
        
        Args:
            image_path (str): Path to image
            
        Returns:
            dict: Prediction result
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        result = self.predictor.predict(image_path)
        self.results.append(result)
        
        return result
    
    def test_directory(self, test_dir):
        """
        Test on all images in directory.
        
        Args:
            test_dir (str): Directory with test images
            
        Returns:
            list: All prediction results
        """
        results = self.predictor.batch_predict(test_dir)
        self.results.extend(results)
        
        return results
    
    def evaluate_predictions(self, ground_truth_dir):
        """
        Evaluate predictions against ground truth.
        
        Ground truth structure:
        ground_truth_dir/
        ├── image1.jpg  (expected to be classified as per filename convention)
        └── ...
        
        Args:
            ground_truth_dir (str): Directory with ground truth labels
            
        Returns:
            dict: Evaluation metrics
        """
        
        y_true = []
        y_pred = []
        
        class_names = []
        
        # Predict on all images
        for img_file in os.listdir(ground_truth_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(ground_truth_dir, img_file)
                
                try:
                    result = self.test_single_image(img_path)
                    y_pred.append(result['class_index'])
                    
                    # Extract class from filename (implementation-specific)
                    # This is a placeholder - adapt based on your naming convention
                    ground_truth_class = img_file.split('_')[0]
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
        
        if not y_true or not y_pred:
            print("⚠️ No predictions made")
            return None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def print_results_summary(self):
        """Print summary of test results."""
        
        if not self.results:
            print("⚠️ No results to display")
            return
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60 + "\n")
        
        total = len(self.results)
        defects = sum(1 for r in self.results if r.get('is_defect'))
        normal = total - defects
        
        print(f"Total Images Tested: {total}")
        print(f"Defects Detected: {defects}")
        print(f"Normal Surfaces: {normal}")
        print(f"Defect Rate: {defects/total*100:.1f}%\n")
        
        # Class distribution
        class_dist = {}
        for result in self.results:
            class_name = result.get('predicted_class', 'Unknown')
            class_dist[class_name] = class_dist.get(class_name, 0) + 1
        
        print("Class Distribution:")
        for class_name in sorted(class_dist.keys()):
            count = class_dist[class_name]
            pct = count / total * 100
            print(f"  {class_name}: {count:3d} ({pct:5.1f}%)")
        
        # Confidence statistics
        confidences = [r.get('confidence', 0) for r in self.results]
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        print(f"\nConfidence Statistics:")
        print(f"  Average: {avg_confidence*100:.2f}%")
        print(f"  Minimum: {min_confidence*100:.2f}%")
        print(f"  Maximum: {max_confidence*100:.2f}%")
        
        print("\n" + "="*60 + "\n")


def run_quick_test():
    """Run a quick test on sample image."""
    
    print("\n" + "🧪 "*20)
    print("Model Quick Test")
    print("🧪 "*20 + "\n")
    
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"❌ Model not found at: {CNN_MODEL_PATH}")
        print("Please train the model first: python train.py")
        return
    
    print("✅ Model loaded successfully!")
    
    # Look for a test image
    test_image = None
    for root, dirs, files in os.walk('data/'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if not test_image:
        print("⚠️ No test image found in data/ directory")
        print("Place an image in data/ folder and try again")
        return
    
    print(f"\n📷 Testing on: {test_image}")
    
    tester = ModelTester()
    result = tester.test_single_image(test_image)
    
    print(f"\n✅ Prediction Results:")
    print(f"   Class: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence_score']}")
    print(f"   Is Defect: {result['is_defect']}")
    
    print(f"\n📊 All Probabilities:")
    for class_name, prob in sorted(result['all_probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"   {class_name}: {prob*100:6.2f}%")


def main():
    """Main testing script."""
    
    print("\n" + "🧪 "*20)
    print("Model Testing & Validation Tool")
    print("🧪 "*20 + "\n")
    
    print("Quick Test on Sample Image")
    run_quick_test()
    
    print("\n" + "="*60)
    print("\nFor detailed testing:")
    print("  1. Place test images in 'test_images/' directory")
    print("  2. Run: python -c \"from test_utils import ModelTester; tester = ModelTester(); results = tester.test_directory('test_images/'); tester.print_results_summary()\"")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
