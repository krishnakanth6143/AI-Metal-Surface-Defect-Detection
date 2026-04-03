# Project Setup Notes

## Installation Status

- [x] Project structure created
- [x] Dependencies defined (requirements.txt)
- [x] Image preprocessing module (OpenCV)
- [x] CNN model architecture
- [x] Training pipeline
- [x] Prediction/inference module
- [x] Flask web application
- [x] Web templates (HTML/CSS/JavaScript)
- [x] Dataset download utility
- [x] Visualization tools
- [x] Testing utilities
- [x] Documentation (README.md)
- [x] Setup guides

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python download_dataset.py

# 3. Train model
python train.py

# 4. Start web interface
python web/app.py

# 5. Access at http://localhost:5000
```

## Project Structure

```
AI-Based Metal Surface Defect Detection/
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Preprocessed images
├── src/
│   ├── config.py          # Configuration
│   ├── preprocessing/     # Image processing
│   └── models/            # CNN architectures
├── web/
│   ├── app.py             # Flask application
│   ├── templates/         # HTML templates
│   └── static/            # CSS/JavaScript
├── models/                # Trained models
├── train.py               # Training script
├── predict.py             # Inference module
├── download_dataset.py    # Dataset downloader
├── visualize.py           # Visualization tools
├── test_utils.py          # Testing utilities
├── example_usage.py       # Examples
├── setup_guide.py         # Setup documentation
└── requirements.txt       # Dependencies
```

## Key Features

✅ **Image Preprocessing**
- Grayscale conversion
- Noise reduction
- Edge detection
- Image resizing and normalization

✅ **Deep Learning Models**
- Custom CNN
- MobileNetV2 (transfer learning)
- ResNet50 (transfer learning)
- Data augmentation

✅ **Web Interface**
- Drag-and-drop upload
- Real-time predictions
- Confidence visualization
- Prediction history
- Session statistics

✅ **Utilities**
- Dataset analysis
- Training visualization
- Performance testing
- Example scripts

## Configuration

Edit `src/config.py` to customize:
- Image size: IMG_SIZE = 224
- Batch size: BATCH_SIZE = 32
- Epochs: EPOCHS = 50
- Learning rate: LEARNING_RATE = 0.001

## Dataset

The project expects the NEU Surface Defect Dataset organized as:

```
data/raw/
├── Normal/          (200+ images)
├── Scratch/         (200+ images)
├── Inclusion/       (200+ images)
├── Pitted_Surface/  (200+ images)
├── Rolled-in_Scale/ (200+ images)
└── Cratering/       (200+ images)
```

Download from: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset: `python download_dataset.py`
3. Train model: `python train.py`
4. Start web app: `python web/app.py`
5. Access: http://localhost:5000

## Troubleshooting

- **Model not found**: Train first with `python train.py`
- **No dataset**: Run `python download_dataset.py`
- **Out of memory**: Reduce BATCH_SIZE in config.py
- **Port in use**: Change port in web/app.py or kill existing process

## Performance Expectations

- Training time: 20-60 minutes (depends on dataset size)
- Inference time: ~50ms per image
- Expected accuracy: 90-97%
- Model size: ~100-300MB

## Support

Refer to README.md for comprehensive documentation.
