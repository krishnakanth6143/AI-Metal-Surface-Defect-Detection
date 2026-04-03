"""
Example usage and testing script.
Demonstrates how to use the defect detection system.
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import ImageProcessor
from predict import DefectPredictor
from src.config import CNN_MODEL_PATH, RAW_DATA_DIR


def example_preprocessing():
    """Example: Image preprocessing."""
    print("\n" + "="*50)
    print("Example 1: Image Preprocessing")
    print("="*50)
    
    # Initialize processor
    processor = ImageProcessor(target_size=(224, 224))
    
    # Example image path (you need to provide an actual image)
    example_image = "example_image.jpg"
    
    if os.path.exists(example_image):
        print(f"✓ Loading image: {example_image}")
        
        # Preprocess single image
        processed = processor.preprocess_image(example_image, normalize=True)
        print(f"✓ Preprocessed image shape: {processed.shape}")
        print(f"✓ Image normalized to range [0, 1]")
        
        # Advanced preprocessing with edge detection
        normalized, edges, resized = processor.preprocess_for_analysis(example_image)
        print(f"✓ Edge detection completed")
        print(f"✓ Original resized shape: {resized.shape}")
        print(f"✓ Edges shape: {edges.shape}")
    else:
        print(f"ℹ To test preprocessing:")
        print(f"  1. Place an image at: {example_image}")
        print(f"  2. Run this script again")


def example_prediction():
    """Example: Make predictions."""
    print("\n" + "="*50)
    print("Example 2: Make Predictions")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"⚠ Model not found at: {CNN_MODEL_PATH}")
        print("✓ To use prediction:")
        print("  1. Run: python train.py")
        print("  2. Then run: python example_usage.py")
        return
    
    # Initialize predictor
    print("✓ Loading model...")
    predictor = DefectPredictor(CNN_MODEL_PATH)
    
    # Example prediction
    example_image = "example_image.jpg"
    
    if os.path.exists(example_image):
        print(f"✓ Predicting on: {example_image}")
        
        result = predictor.predict(example_image)
        
        print(f"\n📊 Prediction Results:")
        print(f"   Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence_score']}")
        print(f"   Is Defect: {result['is_defect']}")
        
        print(f"\n📈 All Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            percentage = f"{prob*100:.2f}%"
            print(f"   {class_name}: {percentage}")
    else:
        print(f"ℹ To test prediction:")
        print(f"  1. Place an image at: {example_image}")
        print(f"  2. Run: python example_usage.py")


def example_batch_processing():
    """Example: Batch processing multiple images."""
    print("\n" + "="*50)
    print("Example 3: Batch Processing")
    print("="*50)
    
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"⚠ Model not found. Please train first: python train.py")
        return
    
    # Initialize predictor
    predictor = DefectPredictor(CNN_MODEL_PATH)
    
    test_dir = "test_images"
    
    if os.path.exists(test_dir):
        print(f"✓ Processing images from: {test_dir}")
        
        results = predictor.batch_predict(test_dir)
        summary = predictor.get_defect_summary(results)
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total Images: {summary['total_images']}")
        print(f"   Defects Found: {summary['defects_found']}")
        print(f"   Normal Surfaces: {summary['normal_surfaces']}")
        print(f"   Defect Rate: {summary['defect_percentage']:.1f}%")
        
        print(f"\n📈 Class Distribution:")
        for class_name, count in summary['class_distribution'].items():
            print(f"   {class_name}: {count}")
    else:
        print(f"ℹ To test batch processing:")
        print(f"  1. Create directory: {test_dir}")
        print(f"  2. Place images inside")
        print(f"  3. Run: python example_usage.py")


def show_menu():
    """Show interactive menu."""
    print("\n" + "="*50)
    print("Metal Surface Defect Detection - Examples")
    print("="*50)
    print("\nChoose an example to run:")
    print("1. Image Preprocessing")
    print("2. Single Image Prediction")
    print("3. Batch Processing")
    print("4. Show All Examples")
    print("0. Exit")
    print("")
    
    choice = input("Enter your choice (0-4): ").strip()
    
    if choice == '1':
        example_preprocessing()
    elif choice == '2':
        example_prediction()
    elif choice == '3':
        example_batch_processing()
    elif choice == '4':
        example_preprocessing()
        example_prediction()
        example_batch_processing()
    elif choice == '0':
        print("👋 Goodbye!")
        return False
    else:
        print("❌ Invalid choice")
    
    return True


def main():
    """Main function."""
    print("\n" + "🔬 "*10)
    print("Metal Surface Defect Detection System")
    print("Example Usage & Testing")
    print("🔬 "*10 + "\n")
    
    print("📋 Quick Start Guide:")
    print("1. Download dataset: python download_dataset.py")
    print("2. Train model: python train.py")
    print("3. Start web UI: python web/app.py")
    print("4. Make predictions: python predict.py")
    
    # Show examples
    while show_menu():
        pass


if __name__ == '__main__':
    main()
