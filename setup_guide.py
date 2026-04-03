"""
Quick setup and first-run guide.
Provides step-by-step instructions to get the system running.
"""

import os
import sys
from pathlib import Path


SETUP_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     AI-BASED METAL SURFACE DEFECT DETECTION SYSTEM                          ║
║     Getting Started Guide                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 SYSTEM REQUIREMENTS
══════════════════════════════════════════════════════════════════════════════

✓ Python 3.8 or higher
✓ 4GB RAM (minimum)
✓ 2GB free disk space
✓ GPU recommended (NVIDIA CUDA) for faster training

📦 INSTALLATION (5 minutes)
══════════════════════════════════════════════════════════════════════════════

Step 1: Install Python Dependencies
────────────────────────────────────
Run in PowerShell or Command Prompt:

    pip install -r requirements.txt

This installs all required packages including:
  - TensorFlow/Keras (Deep Learning)
  - OpenCV (Image Processing)
  - Flask (Web Framework)
  - scikit-learn (ML Utilities)


Step 2: Download Dataset
────────────────────────
Run:
    python download_dataset.py

This will show you how to download the NEU Surface Defect Dataset.

Manual steps:
  1. Visit: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
  2. Download the BMP format dataset
  3. Extract to: data/raw/

Expected structure:
  data/raw/
  ├── Normal/              (200+ BMP images)
  ├── Scratch/             (200+ BMP images)
  ├── Inclusion/           (200+ BMP images)
  ├── Pitted_Surface/      (200+ BMP images)
  ├── Rolled-in_Scale/     (200+ BMP images)
  └── Cratering/           (200+ BMP images)


🚀 QUICK START (Choose one workflow)
══════════════════════════════════════════════════════════════════════════════

WORKFLOW 1: Web Interface (Recommended for First-Time Users)
──────────────────────────────────────────────────────────────

Purpose: Upload images via web browser and get instant predictions

⚠️  IMPORTANT: Skip this step if you don't have a pre-trained model yet!
   First, complete "WORKFLOW 3: Train Model" below

When ready:
  1. Run: python web/app.py
  2. Open browser: http://localhost:5000
  3. Upload images via drag-and-drop
  4. View results instantly with confidence scores


WORKFLOW 2: Command-Line Predictions
──────────────────────────────────────

Purpose: Make predictions from terminal

  Options:
  a) Single image prediction:
     Run: python -c "from predict import predict_single_image; print(predict_single_image('image.jpg'))"

  b) Interactive mode:
     Run: python predict.py

  c) Test utility:
     Run: python test_utils.py


WORKFLOW 3: Train Model (Required if you don't have models/defect_cnn_model.h5)
────────────────────────────────────────────────────────────────────────────────

Purpose: Train a new CNN model on your dataset

Requirements:
  ✓ Dataset installed in data/raw/
  ✓ At least 600 images total (100+ per class)
  ✓ 2GB RAM and 10 min - 1 hour (depends on dataset size)

Run:
  1. python train.py

Monitor:
  - Watch the console for training progress
  - Training metrics update each epoch:
    * Training Accuracy
    * Validation Accuracy
    * Model Loss

Output:
  ✓ models/defect_cnn_model.h5  (trained model)
  ✓ training_info.json           (metadata)

After training completes:
  - Best validation accuracy will be displayed
  - Test accuracy will be shown
  - Model is ready for predictions!


WORKFLOW 4: Analyze Dataset
─────────────────────────────

Purpose: Visualize dataset distribution and inspect images

Run:
  python visualize.py

Options:
  1. Show class distribution
  2. Plot statistics
  3. View sample images
  4. Generate analysis report


WORKFLOW 5: Full Automation
────────────────────────────

Purpose: Run complete pipeline end-to-end

Run:
  python example_usage.py

Provides:
  - Interactive menu
  - Preprocessing examples
  - Training & prediction examples
  - Batch processing examples


📊 UNDERSTANDING THE OUTPUT
══════════════════════════════════════════════════════════════════════════════

Training Output Example:
────────────────────────
    Epoch 1/50
    120/120 [==============================] - loss: 1.2345 - accuracy: 0.6234
    val_loss: 0.8765 - val_accuracy: 0.7523

    Epoch 50/50
    120/120 [==============================] - loss: 0.0234 - accuracy: 0.9854
    val_loss: 0.1456 - val_accuracy: 0.9521


Prediction Output Example:
──────────────────────────
    {
      'predicted_class': 'Scratch',
      'confidence': 0.962,
      'is_defect': True,
      'all_probabilities': {
        'Normal': 0.001,
        'Scratch': 0.962,
        'Inclusion': 0.015,
        'Pitted_Surface': 0.012,
        'Rolled-in_Scale': 0.008,
        'Cratering': 0.002
      }
    }

Interpretation:
  - Model is 96.2% confident this is a scratch defect
  - Quality control recommendation: REJECT this part


⚙️  CONFIGURATION (Optional - Advanced)
══════════════════════════════════════════════════════════════════════════════

Edit src/config.py to modify:

Image Settings:
  IMG_SIZE = 224                    # Input image size (pixels)

Training Settings:
  BATCH_SIZE = 32                   # Images per batch
  EPOCHS = 50                       # Training epochs
  LEARNING_RATE = 0.001             # Optimizer learning rate

Dataset Split:
  TEST_SIZE = 0.2                   # 20% for testing
  VALIDATION_SIZE = 0.1             # 10% for validation

Defect Classes:
  DEFECT_CLASSES = {                # Can customize classes
    0: 'Normal',
    1: 'Scratch',
    ...
  }


🧪 TESTING & VALIDATION
══════════════════════════════════════════════════════════════════════════════

Test on Single Image:
  python -c "from test_utils import ModelTester; tester = ModelTester(); result = tester.test_single_image('path/to/image.jpg'); print(result)"

Test on Directory:
  python -c "from test_utils import ModelTester; tester = ModelTester(); results = tester.test_directory('test_images/'); tester.print_results_summary()"

Visualize Dataset:
  python visualize.py

Analyze Results:
  python example_usage.py


🐛 TROUBLESHOOTING
══════════════════════════════════════════════════════════════════════════════

Problem: "Model not found"
Solution:
  ✓ Check if models/ directory exists
  ✓ Train model first: python train.py
  ✓ Verify model file: models/defect_cnn_model.h5

Problem: "No training data found"
Solution:
  ✓ Download dataset: python download_dataset.py
  ✓ Check data/raw/ contains class subdirectories
  ✓ Verify images are in correct format (BMP, JPG, PNG)

Problem: "Out of Memory (OOM) Error"
Solution:
  ✓ Reduce BATCH_SIZE in src/config.py
  ✓ Close other applications
  ✓ Use GPU if available

Problem: "Slow Training"
Solution:
  ✓ Reduce BATCH_SIZE
  ✓ Use MobileNetV2 model (faster)
  ✓ Enable GPU acceleration

Problem: "Web interface won't open"
Solution:
  ✓ Ensure Flask is installed: pip install Flask
  ✓ Check port 5000 is not in use
  ✓ Try port 5001: edit web/app.py line: app.run(port=5001)

Problem: "Unable to import TensorFlow"
Solution:
  ✓ Reinstall TensorFlow: pip install --upgrade tensorflow
  ✓ Verify Python version (3.8+): python --version
  ✓ Check pip installation: pip list | grep tensorflow


📁 FILE REFERENCE
══════════════════════════════════════════════════════════════════════════════

Utility Scripts:
  download_dataset.py      - Download and setup dataset
  train.py                 - Train CNN model
  predict.py               - Initialize predictor and test inference
  visualize.py             - Analyze dataset and plot statistics
  test_utils.py            - Test model and evaluate performance
  example_usage.py         - Interactive examples

Core Modules:
  src/config.py            - Configuration and paths
  src/preprocessing/       - Image preprocessing
  src/models/              - CNN architectures

Web Application:
  web/app.py               - Flask application
  web/templates/index.html - Web interface
  web/static/              - CSS and JavaScript

Data:
  data/raw/                - Original dataset (organize by class)
  data/processed/          - Preprocessed images
  models/                  - Trained model storage


💡 NEXT STEPS
══════════════════════════════════════════════════════════════════════════════

1. ✓ Install dependencies: pip install -r requirements.txt
2. ✓ Download dataset: python download_dataset.py
3. ✓ Train model: python train.py (30min - 1 hour)
4. ✓ Test predictions: python predict.py
5. ✓ Launch web UI: python web/app.py
6. ✓ Analyze results: python visualize.py

Advanced:
  - Customize model architecture
  - Collect more training data
  - Fine-tune hyperparameters
  - Deploy to production
  - Integrate with manufacturing systems


📞 SUPPORT & RESOURCES
══════════════════════════════════════════════════════════════════════════════

Documentation: See README.md for detailed documentation

TensorFlow: https://www.tensorflow.org/
OpenCV: https://opencv.org/
Flask: https://flask.palletsprojects.com/

Dataset Info: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

Kaggle Datasets: https://www.kaggle.com/datasets/
  - NEU Surface Defect Database
  - Other metal surface defect datasets


═══════════════════════════════════════════════════════════════════════════════
Ready to start? Run: python train.py
═══════════════════════════════════════════════════════════════════════════════
"""


def show_setup_guide():
    """Display the complete setup guide."""
    print(SETUP_GUIDE)


def create_checklist():
    """Create setup checklist."""
    
    checklist = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SETUP CHECKLIST                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALLATION
  ☐ Python 3.8+ installed
  ☐ Ran: pip install -r requirements.txt
  ☐ All packages installed successfully

DATASET
  ☐ Downloaded NEU Surface Defect Dataset
  ☐ Extracted to: data/raw/
  ☐ Organized by class (Normal, Scratch, etc.)
  ☐ Verified 6 class subdirectories exist
  ☐ Verified 100+ images per class

MODEL TRAINING
  ☐ Ran: python train.py
  ☐ Training completed without errors
  ☐ Model saved to: models/defect_cnn_model.h5
  ☐ Noted training and test accuracy

TESTING
  ☐ Ran: python predict.py
  ☐ Made prediction on test image
  ☐ Verified prediction confidence scores

WEB INTERFACE
  ☐ Ran: python web/app.py
  ☐ Opened: http://localhost:5000
  ☐ Uploaded test image
  ☐ Verified prediction displayed correctly

ANALYSIS & VISUALIZATION
  ☐ Ran: python visualize.py
  ☐ Viewed dataset statistics
  ☐ Reviewed confusion matrix / training history

DEPLOYMENT READY
  ☐ All tests passed
  ☐ Model performance acceptable (>90% accuracy)
  ☐ Web interface working correctly
  ☐ Data pipeline tested end-to-end

═══════════════════════════════════════════════════════════════════════════════
"""
    
    print(checklist)


def main():
    """Main setup guide."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'checklist':
            create_checklist()
            return
    
    print("\n" + "🚀 "*40 + "\n")
    show_setup_guide()
    print("\n" + "🚀 "*40 + "\n")
    
    print("📋 Run checklist: python setup_guide.py checklist\n")


if __name__ == '__main__':
    main()
