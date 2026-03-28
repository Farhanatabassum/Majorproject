from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json

app = Flask(__name__)
app.secret_key = 'radioactive_watermark_detector_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        # Simulate training for now
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully (simulated)',
            'metrics': {
                'accuracy': 0.95,
                'precision': 0.93,
                'recall': 0.97,
                'f1_score': 0.95
            },
            'dataset_info': {
                'total_images': 100,
                'watermarked': 50,
                'original': 50
            }
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
    
    try:
        # Simulate detection for now
        import random
        prediction = random.choice([0, 1])
        probability = random.uniform(0.7, 0.98)
        
        result = {
            'prediction': prediction,
            'probability': probability,
            'label': 'Watermarked' if prediction == 1 else 'Original',
            'confidence': probability * 100 if prediction == 1 else (1 - probability) * 100
        }
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'model_loaded': True,
        'model_exists': True,
        'dataset_available': True,
        'device': 'CPU (Simulated)'
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
    
    # Run the Flask app
    print("🚀 Starting Radioactive Watermark Detection System...")
    print("📱 Web interface will be available at: http://localhost:5000")
    print("⚠️  Note: This is a demo version with simulated results")
    print("🔧 Full PyTorch integration requires additional setup")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
