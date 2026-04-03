"""
Flask web application for Metal Surface Defect Detection.
Provides web interface for uploading images and displaying predictions.
"""

import os
import sys
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from predict import DefectPredictor


# Initialize Flask app
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize predictor
try:
    print(f"[INFO] Attempting to load model from: {CNN_MODEL_PATH}")
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"[ERROR] Model file not found: {CNN_MODEL_PATH}")
        predictor = None
        MODEL_LOADED = False
    else:
        print(f"[INFO] Model file exists, loading...")
        predictor = DefectPredictor(CNN_MODEL_PATH)
        MODEL_LOADED = True
        print(f"[SUCCESS] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {str(e)}")
    import traceback
    traceback.print_exc()
    predictor = None
    MODEL_LOADED = False

# Store prediction history
prediction_history = []


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_defect_info(image, prediction_result):
    """
    Draw prediction info on image.
    
    Args:
        image (numpy.ndarray): Image array
        prediction_result (dict): Prediction result
        
    Returns:
        numpy.ndarray: Image with annotations
    """
    image_copy = image.copy()
    h, w = image_copy.shape[:2]
    
    # Convert to BGR for OpenCV
    if len(image_copy.shape) == 3 and image_copy.shape[2] == 3:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
    
    # Draw border and text
    color = (0, 0, 255) if prediction_result['is_defect'] else (0, 255, 0)
    thickness = 3
    
    cv2.rectangle(image_copy, (10, 10), (w-10, h-10), color, thickness)
    
    # Add text
    text = f"{prediction_result['predicted_class']}: {prediction_result['confidence_score']}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    # Background for text
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x, text_y = 20, 50
    
    cv2.rectangle(image_copy, (text_x - 5, text_y - 30), 
                 (text_x + text_size[0] + 5, text_y + 5), 
                 (255, 255, 255), -1)
    cv2.putText(image_copy, text, (text_x, text_y), font, font_scale, color, font_thickness)
    
    return image_copy


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', model_loaded=MODEL_LOADED)


def process_uploaded_image(file):
    """Save uploaded file, run prediction, and store history."""
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    prediction = predictor.predict(filepath)
    prediction['filename'] = filename
    prediction['timestamp'] = datetime.now().isoformat()
    prediction['filepath'] = filepath
    prediction_history.append(prediction)

    return filename, prediction


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction."""
    
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 400
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400
    
    try:
        filename, prediction = process_uploaded_image(file)
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'prediction': {
                'class': prediction['predicted_class'],
                'confidence': prediction['confidence_score'],
                'is_defect': prediction['is_defect'],
                'probabilities': prediction['all_probabilities']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_file():
    """Compatibility endpoint for the new index.html design."""

    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 400

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400

    try:
        _, prediction = process_uploaded_image(file)
        
        # Debug: Print prediction result
        print(f"[DEBUG] Prediction result: {prediction}")
        
        # Prepare probabilities - ensure it's a proper dict
        probabilities = prediction.get('all_probabilities', {})
        print(f"[DEBUG] Probabilities: {probabilities}")

        # Keep these keys for the new template's inline JS.
        response = {
            'success': True,
            'class': prediction['predicted_class'],
            'predicted_class': prediction['predicted_class'],
            'confidence': float(prediction['confidence']),
            'is_defect': bool(prediction['is_defect']),
            'probabilities': probabilities,
            'confidence_score': prediction.get('confidence_score', f"{float(prediction['confidence']) * 100:.2f}%")
        }
        
        print(f"[DEBUG] Response: {response}")
        
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze/<filename>')
def analyze_image(filename):
    """Analyze image and return detailed results."""
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Load and analyze
        prediction = predictor.predict(filepath)
        
        # Load image for preview
        image = cv2.imread(filepath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _, image_encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_base64 = 'data:image/jpeg;base64,' + \
                         __import__('base64').b64encode(image_encoded).decode()
        else:
            image_base64 = None
        
        response = {
            'success': True,
            'analysis': {
                'predicted_class': prediction['predicted_class'],
                'confidence': prediction['confidence_score'],
                'is_defect': prediction['is_defect'],
                'probabilities': prediction['all_probabilities'],
                'recommendation': 'REJECT' if prediction['is_defect'] else 'ACCEPT'
            },
            'image': image_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history')
def get_history():
    """Get prediction history."""
    summary = {
        'total_predictions': len(prediction_history),
        'defects_found': sum(1 for p in prediction_history if p['is_defect']),
        'normal_surfaces': sum(1 for p in prediction_history if not p['is_defect']),
    }
    
    if summary['total_predictions'] > 0:
        summary['defect_rate'] = f"{summary['defects_found'] / summary['total_predictions'] * 100:.1f}%"
    
    return jsonify({
        'success': True,
        'summary': summary,
        'history': prediction_history[-20:]  # Last 20 predictions
    })


@app.route('/stats')
def get_stats():
    """Get statistics."""
    
    class_distribution = {}
    for prediction in prediction_history:
        class_name = prediction['predicted_class']
        class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
    
    stats = {
        'total_predictions': len(prediction_history),
        'defects_found': sum(1 for p in prediction_history if p['is_defect']),
        'class_distribution': class_distribution,
        'avg_confidence': np.mean([p['confidence'] for p in prediction_history]) if prediction_history else 0
    }
    
    return jsonify({
        'success': True,
        'stats': stats
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'predictions_made': len(prediction_history)
    })


@app.errorhandler(400)
def bad_request(e):
    """Handle 400 errors."""
    return jsonify({'success': False, 'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'success': False, 'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("=" * 50)
    print("Metal Surface Defect Detection - Web App")
    print("=" * 50)
    print(f"Model loaded: {MODEL_LOADED}")
    print("🚀 Starting server at http://localhost:5000")
    print("=" * 50)
    
    # Disable debug mode to prevent reload issues with model loading
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
