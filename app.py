from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
from radioactive_watermark_detector import RadioactiveWatermarkDetector, CNNModel
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'radioactive_watermark_detector_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MODEL_PATH = 'processed_dataset/watermark_detector_model.pth'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
detector = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, detector
    
    try:
        model = CNNModel().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully")
        return True
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess image for model input - Updated for new model"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Updated to match new model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict_watermark(image_path):
    """Predict if image contains watermark"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        image_tensor = preprocess_image(image_path)
        
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
            
        return {
            'prediction': prediction,
            'probability': probability,
            'label': 'Watermarked' if prediction == 1 else 'Original',
            'confidence': probability * 100 if prediction == 1 else (1 - probability) * 100
        }, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/train')
def train_page():
    """Training page"""
    return render_template('train.html')

@app.route('/detect')
def detect_page():
    """Detection page"""
    return render_template('detect.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    """API endpoint to train the model"""
    try:
        # Get training parameters from request
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Extract parameters with defaults (since form fields are removed)
        sample_size = int(data.get('sampleSize', 5))
        replace_count = int(data.get('replaceCount', 1))
        epochs = 1  # Single epoch for instant training
        batch_size = 256  # Maximum batch size for speed
        watermark_type = data.get('watermarkType', 'noise')
        learning_rate = float(data.get('learningRate', 0.001))
        datasets = data.get('datasets', ['mirflickr'])  # Support multiple datasets
        
        # Initialize detector with first dataset (or default)
        detector = RadioactiveWatermarkDetector(dataset_path=datasets[0])
        
        # Run the complete training pipeline with user parameters
        print(f"Starting training with parameters: sample_size={sample_size}, replace_count={replace_count}, epochs={epochs}")
        print(f"Using datasets: {datasets}")
        
        # Load images from all datasets
        all_image_files = []
        for dataset_path in datasets:
            detector.dataset_path = dataset_path
            image_files = detector.load_dataset()
            all_image_files.extend(image_files)
            print(f"Loaded {len(image_files)} images from {dataset_path}")
        
        print(f"Total images from all datasets: {len(all_image_files)}")
        
        # Use the combined dataset for sampling and processing
        sampled_files, remaining_files = detector.random_sampling(all_image_files, sample_size=sample_size)
        watermarked_files = detector.create_watermarked_dataset(sampled_files)
        final_watermarked, final_originals = detector.replace_watermarked_images(
            watermarked_files, remaining_files, replace_count=replace_count
        )
        
        # Update detector output path to use the current working directory
        detector.output_path = "processed_dataset"  # Use relative path in working directory
        
        # Prepare and train
        image_paths, labels = detector.prepare_dataset_for_training()
        train_loader, test_loader = detector.create_data_loaders(image_paths, labels, test_size=0.2, batch_size=batch_size)
        model, train_losses, test_losses, train_accuracies, test_accuracies = detector.train_model(
            train_loader, test_loader, epochs=epochs, learning_rate=learning_rate
        )
        
        metrics = detector.evaluate_model(model, test_loader)
        
        print(f"Training completed successfully! Metrics: {metrics}")
        
        # Get actual processed dataset counts from the correct detector output path
        watermarked_dir = os.path.join(detector.output_path, "mixed_dataset")
        original_dir = os.path.join(detector.output_path, "original")
        
        # Check if directories exist before counting
        watermarked_count = 0
        original_count = 0
        
        if os.path.exists(watermarked_dir):
            watermarked_count = len([f for f in os.listdir(watermarked_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        if os.path.exists(original_dir):
            original_count = len([f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        print(f"Actual processed counts - Watermarked: {watermarked_count}, Original: {original_count}")
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': metrics,
            'dataset_info': {
                'total_images': len(image_paths),
                'watermarked': watermarked_count,
                'original': original_count,
                'datasets_used': datasets,
                'images_per_dataset': {ds: len([f for f in all_image_files if ds in f]) for ds in datasets}
            },
            'parameters_used': {
                'sample_size': sample_size,
                'replace_count': replace_count,
                'epochs': epochs,
                'batch_size': batch_size,
                'watermark_type': watermark_type,
                'learning_rate': learning_rate
            }
        })
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/select-dataset', methods=['POST'])
def select_dataset():
    """API endpoint to select custom dataset folder"""
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        dataset_path = data.get('datasetPath', 'mirflickr')
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            return jsonify({
                'status': 'error',
                'message': f'Dataset path does not exist: {dataset_path}'
            }), 400
        
        # Count images in dataset
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        image_count = 0
        for file in os.listdir(dataset_path):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                image_count += 1
        
        return jsonify({
            'status': 'success',
            'message': f'Dataset selected successfully',
            'dataset_path': dataset_path,
            'image_count': image_count,
            'supported_formats': supported_formats
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/detect', methods=['POST'])
def detect_watermark():
    """API endpoint to detect watermark in uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Load model if not already loaded
    if model is None:
        if not load_model():
            return jsonify({'error': 'Model not available. Please train the model first.'}), 500
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Make prediction
        result, error = predict_watermark(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    model_loaded = model is not None
    model_exists = os.path.exists(MODEL_PATH)
    dataset_exists = os.path.exists('mirflickr') and len(os.listdir('mirflickr')) > 0
    
    return jsonify({
        'model_loaded': model_loaded,
        'model_exists': model_exists,
        'dataset_available': dataset_exists,
        'device': str(device)
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dataset/<path:filename>')
def serve_dataset_file(filename):
    """Serve dataset files for preview"""
    return send_from_directory('mirflickr', filename)

@app.route('/processed/<path:filename>')
def serve_processed_file(filename):
    """Serve processed dataset files"""
    return send_from_directory('processed_dataset', filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Try to load the model
    load_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
