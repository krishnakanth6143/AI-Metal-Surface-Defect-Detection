"""
Training pipeline for Metal Surface Defect Detection model.
Handles data loading, preprocessing, model training, and evaluation.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.preprocessing import ImageProcessor
from src.models import DefectCNN


class DataLoader:
    """Handles data loading and preprocessing."""
    
    def __init__(self, data_dir, img_size=IMG_SIZE):
        """
        Initialize data loader.
        
        Args:
            data_dir (str): Directory containing class subdirectories
            img_size (int): Target image size
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.processor = ImageProcessor(target_size=(img_size, img_size))
        self.class_names = []
        self.class_indices = {}
    
    def load_data_from_directory(self):
        """
        Load images from directory structure with class subdirectories.
        
        Directory structure expected:
        data/
        ├── Normal/
        ├── Scratch/
        ├── Inclusion/
        └── ...
        
        Returns:
            tuple: (images, labels, class_names)
        """
        images = []
        labels = []
        
        class_dirs = sorted([d for d in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, d))])
        
        self.class_names = class_dirs
        self.class_indices = {name: idx for idx, name in enumerate(class_dirs)}
        
        print(f"Classes found: {self.class_names}")
        
        for class_name in class_dirs:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_indices[class_name]
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    image = self.processor.preprocess_image(img_path)
                    images.append(image)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels), self.class_names
    
    def create_data_generators(self):
        """
        Create data augmentation generators.
        
        Returns:
            keras.preprocessing.image.ImageDataGenerator: Generator object
        """
        train_generator = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        validation_generator = keras.preprocessing.image.ImageDataGenerator()
        
        return train_generator, validation_generator


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model=None, model_type='custom'):
        """
        Initialize model trainer.
        
        Args:
            model (keras.Model): Pre-built model or None
            model_type (str): 'custom', 'mobilenet', or 'resnet'
        """
        self.model = model
        self.model_type = model_type
        self.history = None
        self.best_val_accuracy = 0
    
    def build_model(self, img_size=IMG_SIZE, num_classes=6):
        """
        Build model if not already provided.
        
        Args:
            img_size (int): Input image size
            num_classes (int): Number of classes
        """
        if self.model_type == 'custom':
            self.model = DefectCNN.build_model(img_size, num_classes)
        elif self.model_type == 'mobilenet':
            self.model = DefectCNN.build_mobilenet_model(img_size, num_classes)
        elif self.model_type == 'resnet':
            self.model = DefectCNN.build_resnet_model(img_size, num_classes)
        
        self.model = DefectCNN.compile_model(self.model, learning_rate=LEARNING_RATE)
        print(f"Built {self.model_type} model")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, 
              batch_size=BATCH_SIZE, model_path=CNN_MODEL_PATH):
        """
        Train the model.
        
        Args:
            X_train (numpy.ndarray): Training images
            y_train (numpy.ndarray): Training labels (one-hot encoded)
            X_val (numpy.ndarray): Validation images
            y_val (numpy.ndarray): Validation labels (one-hot encoded)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            model_path (str): Path to save the model
        """
        callbacks = DefectCNN.create_callbacks(model_path)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Training completed. Best validation accuracy: {max(self.history.history['val_accuracy']):.4f}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        
        Args:
            X_test (numpy.ndarray): Test images
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def save_model(self, model_path):
        """Save model to file."""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def save_training_info(self, class_names, info_path='training_info.json'):
        """Save training information."""
        info = {
            'model_type': self.model_type,
            'class_names': class_names,
            'training_date': datetime.now().isoformat(),
            'num_classes': len(class_names)
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"Training info saved to {info_path}")


def prepare_dataset(images, labels, num_classes):
    """
    Prepare dataset for training.
    
    Args:
        images (numpy.ndarray): Image array
        labels (numpy.ndarray): Label array
        num_classes (int): Number of classes
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # One-hot encode labels
    labels_encoded = keras.utils.to_categorical(labels, num_classes)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Main training script."""
    
    print("=" * 50)
    print("Metal Surface Defect Detection - Training Pipeline")
    print("=" * 50)
    
    # Check if data exists
    if not os.path.exists(RAW_DATA_DIR) or len(os.listdir(RAW_DATA_DIR)) == 0:
        print("\n⚠️  No training data found!")
        print(f"Please download the NEU Surface Defect Dataset and place it in: {RAW_DATA_DIR}")
        print("\nExpected structure:")
        print(f"{RAW_DATA_DIR}/")
        print("├── Normal/")
        print("├── Scratch/")
        print("├── Inclusion/")
        print("├── Pitted_Surface/")
        print("├── Rolled-in_Scale/")
        print("└── Cratering/")
        return
    
    # Load data
    print("\n📂 Loading data...")
    data_loader = DataLoader(RAW_DATA_DIR, img_size=IMG_SIZE)
    images, labels, class_names = data_loader.load_data_from_directory()
    
    print(f"✅ Loaded {len(images)} images from {len(class_names)} classes")
    
    # Prepare dataset
    print("\n📊 Preparing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(
        images, labels, len(class_names)
    )
    
    # Build and train model
    print("\n🧠 Building CNN model...")
    trainer = ModelTrainer(model_type='custom')
    trainer.build_model(img_size=IMG_SIZE, num_classes=len(class_names))
    
    print("\n🚀 Training model...")
    trainer.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, 
                 batch_size=BATCH_SIZE, model_path=CNN_MODEL_PATH)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model(CNN_MODEL_PATH)
    trainer.save_training_info(class_names)
    
    print("\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()
