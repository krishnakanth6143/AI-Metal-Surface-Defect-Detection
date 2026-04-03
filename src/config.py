"""
Configuration module for the Metal Surface Defect Detection project.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Defect Classes
DEFECT_CLASSES = {
    0: 'Normal',
    1: 'Scratch',
    2: 'Inclusion',
    3: 'Pitted Surface',
    4: 'Rolled-in Scale',
    5: 'Cratering',
    6: 'Punching'
}

# Model paths
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'defect_cnn_model.h5')
CNN_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'defect_cnn_weights.h5')

# Flask Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'web', 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
