"""
Convolutional Neural Network for Metal Surface Defect Detection.
Implements a CNN architecture for binary and multi-class classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
import numpy as np


class DefectCNN:
    """Custom CNN model for defect detection."""
    
    @staticmethod
    def build_model(img_size=224, num_classes=6):
        """
        Build a custom CNN model.
        
        Args:
            img_size (int): Input image size (square images)
            num_classes (int): Number of output classes
            
        Returns:
            keras.Model: Compiled model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(img_size, img_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def build_mobilenet_model(img_size=224, num_classes=6):
        """
        Build MobileNetV2 transfer learning model (faster, lighter).
        
        Args:
            img_size (int): Input image size
            num_classes (int): Number of output classes
            
        Returns:
            keras.Model: Compiled model
        """
        base_model = MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=(img_size, img_size, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def build_resnet_model(img_size=224, num_classes=6):
        """
        Build ResNet50 transfer learning model (more accurate).
        
        Args:
            img_size (int): Input image size
            num_classes (int): Number of output classes
            
        Returns:
            keras.Model: Compiled model
        """
        base_model = ResNet50(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=(img_size, img_size, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            model (keras.Model): Model to compile
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled model
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model
    
    @staticmethod
    def create_callbacks(model_path, patience=10):
        """
        Create training callbacks for model optimization.
        
        Args:
            model_path (str): Path to save the best model
            patience (int): Patience for early stopping
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        return callbacks
