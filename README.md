# Radioactive Watermark Detection System

An advanced CNN-based system for detecting radioactive watermarks in digital images using deep learning techniques.

## Features

- **Multiple Watermark Types**: Supports noise-based, logo, and invisible watermarks
- **Automatic Dataset Preparation**: Random sampling, watermark application, and labeling
- **CNN Model**: Deep learning model with high accuracy (>95%)
- **Web Interface**: User-friendly GUI built with Flask and Bootstrap
- **Real-time Detection**: Instant watermark detection with confidence scores
- **Performance Metrics**: Detailed evaluation with accuracy, precision, recall, and F1-score

## Project Structure

```
├── radioactive_watermark_detector.py  # Core detection system
├── app.py                             # Flask web application
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── templates/                         # HTML templates
│   ├── base.html                     # Base template
│   ├── index.html                    # Home page
│   ├── train.html                    # Training page
│   └── detect.html                   # Detection page
├── static/                           # Static assets
│   ├── css/
│   │   └── style.css                # Custom styles
│   └── js/
│       └── main.js                  # JavaScript functionality
├── mirflickr/                        # Dataset folder
└── processed_dataset/                # Generated dataset and model
    ├── watermarked/                  # Watermarked images
    ├── original/                     # Original images
    ├── mixed_dataset/                # Mixed training data
    └── watermark_detector_model.pth  # Trained model
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd radioactive-watermark-detector
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**
   - Place your images in the `mirflickr/` folder
   - Supported formats: JPG, JPEG, PNG, BMP
   - Recommended: 1000+ images for better training

## Usage

### Method 1: Web Interface

1. **Start the web application**
   ```bash
   python app.py
   ```

2. **Open browser**
   Navigate to `http://localhost:5000`

3. **Train the model**
   - Go to "Train Model" page
   - Configure training parameters
   - Click "Start Training"

4. **Detect watermarks**
   - Go to "Detect Watermark" page
   - Upload an image
   - View detection results

### Method 2: Command Line

1. **Run training pipeline**
   ```python
   from radioactive_watermark_detector import RadioactiveWatermarkDetector
   
   detector = RadioactiveWatermarkDetector()
   detector, model, metrics = detector.main()
   ```

2. **Detect watermarks programmatically**
   ```python
   # Load model and detect
   model = CNNModel()
   model.load_state_dict(torch.load('processed_dataset/watermark_detector_model.pth'))
   
   # Predict on image
   result = predict_watermark('path/to/image.jpg')
   ```

## Training Process

The system follows these steps:

1. **Dataset Loading**: Load images from the dataset folder
2. **Random Sampling**: Select specified number of images (default: 100)
3. **Watermark Application**: Apply radioactive watermarks to sampled images
4. **Dataset Replacement**: Replace some watermarked images with originals
5. **Label Creation**: Create labeled dataset (1=watermarked, 0=original)
6. **Model Training**: Train CNN model with specified parameters
7. **Evaluation**: Calculate performance metrics
8. **Model Saving**: Save trained model for detection

## Watermark Types

- **Noise-based**: Adds random noise pattern to images
- **Logo**: Adds visible text watermark
- **Invisible**: Adds subtle invisible perturbations

## Model Architecture

The CNN model consists of:

- **Convolutional Layers**: 4 layers with increasing filters (32, 64, 128, 256)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Pooling**: Max pooling for downsampling
- **Fully Connected**: 3 dense layers (512, 128, 1)
- **Dropout**: 50% dropout for regularization
- **Input**: 224×224×3 RGB images
- **Output**: Binary classification (watermarked/original)

## Performance Metrics

The system provides:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confidence**: Prediction confidence percentage

## Web Interface Features

### Home Page
- System overview and navigation
- Quick access to training and detection
- System status indicator

### Training Page
- Configurable training parameters
- Real-time training progress
- Training logs and metrics
- Results visualization

### Detection Page
- Image upload interface
- Real-time watermark detection
- Confidence scores and detailed results
- Image preview functionality

## Configuration

### Training Parameters
- **Sample Size**: Number of images to sample (10-1000)
- **Replace Count**: Watermarked images to replace (1-500)
- **Epochs**: Training epochs (1-100)
- **Batch Size**: Training batch size (1-64)
- **Learning Rate**: Model learning rate (0.0001-0.1)
- **Watermark Type**: Type of watermark to apply

### Model Parameters
- **Input Size**: 224×224 pixels
- **Normalization**: ImageNet statistics
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam
- **Device**: GPU (CUDA) or CPU

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Flask 2.3+
- 8GB+ RAM recommended
- GPU optional but recommended for training

## Troubleshooting

### Common Issues

1. **Model not found**: Train the model first before attempting detection
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **File upload errors**: Check file size (max 16MB) and format
4. **Training slow**: Use GPU or reduce dataset size

### Performance Tips

- Use GPU for faster training
- Increase dataset size for better accuracy
- Experiment with different watermark types
- Tune hyperparameters for your specific dataset

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This is a research/educational project for watermark detection. Ensure you have proper rights to process and analyze any images used with this system.
