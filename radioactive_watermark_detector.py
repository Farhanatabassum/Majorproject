import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
import json

class RadioactiveWatermarkDetector:
    def __init__(self, dataset_path="mirflickr", output_path="processed_dataset"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.create_output_structure()
        
    def create_output_structure(self):
        """Create necessary folder structure for processed dataset"""
        dirs = [
            os.path.join(self.output_path, "watermarked"),
            os.path.join(self.output_path, "original"),
            os.path.join(self.output_path, "mixed_dataset")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def load_dataset(self):
        """Load all images from dataset path"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(self.dataset_path):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                image_files.append(os.path.join(self.dataset_path, file))
        
        print(f"Found {len(image_files)} images in dataset")
        return image_files
    
    def random_sampling(self, image_files, sample_size=100):
        """Randomly select sample_size images from dataset"""
        if len(image_files) < sample_size:
            print(f"Warning: Only {len(image_files)} images available, using all of them")
            sample_size = len(image_files)
        
        sampled_files = random.sample(image_files, sample_size)
        remaining_files = [img for img in image_files if img not in sampled_files]
        
        print(f"Sampled {len(sampled_files)} images")
        print(f"Remaining {len(remaining_files)} images")
        
        return sampled_files, remaining_files
    
    def apply_radioactive_watermark(self, image_path, output_path, watermark_type="noise"):
        """Apply radioactive watermark to an image"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return False
        
        if watermark_type == "noise":
            # Add random noise as watermark
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            watermarked = cv2.add(img, noise)
            
        elif watermark_type == "logo":
            # Add a simple text watermark
            h, w = img.shape[:2]
            cv2.putText(img, 'RADIOACTIVE', (w//4, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            watermarked = img
            
        elif watermark_type == "invisible":
            # Add subtle invisible watermark
            alpha = 0.05
            watermark_pattern = np.random.randint(0, 256, img.shape).astype(np.uint8)
            watermarked = cv2.addWeighted(img, 1-alpha, watermark_pattern, alpha, 0)
        else:
            watermarked = img
        
        # Save watermarked image
        filename = os.path.basename(image_path)
        output_file = os.path.join(output_path, f"wm_{filename}")
        cv2.imwrite(output_file, watermarked)
        
        return True
    
    def create_watermarked_dataset(self, sampled_files):
        """Apply watermarks to all sampled files"""
        watermarked_files = []
        
        for img_path in sampled_files:
            success = self.apply_radioactive_watermark(
                img_path, 
                os.path.join(self.output_path, "watermarked")
            )
            if success:
                watermarked_files.append(img_path)
        
        print(f"Created {len(watermarked_files)} watermarked images")
        return watermarked_files
    
    def replace_watermarked_images(self, watermarked_files, remaining_files, replace_count=50):
        """Replace specified number of watermarked images with original images"""
        if len(watermarked_files) < replace_count:
            replace_count = len(watermarked_files)
        
        if len(remaining_files) < replace_count:
            print(f"Warning: Only {len(remaining_files)} original images available for replacement")
            replace_count = len(remaining_files)
        
        # Randomly select watermarked images to replace
        to_replace = random.sample(watermarked_files, replace_count)
        replacement_originals = random.sample(remaining_files, replace_count)
        
        # Copy original images to mixed dataset
        for orig_path in replacement_originals:
            filename = os.path.basename(orig_path)
            dest_path = os.path.join(self.output_path, "original", filename)
            shutil.copy2(orig_path, dest_path)
        
        # Copy remaining watermarked images to mixed dataset
        remaining_watermarked = [img for img in watermarked_files if img not in to_replace]
        for wm_path in remaining_watermarked:
            filename = os.path.basename(wm_path)
            src_path = os.path.join(self.output_path, "watermarked", f"wm_{filename}")
            dest_path = os.path.join(self.output_path, "mixed_dataset", f"wm_{filename}")
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
        
        # Copy replacement originals to mixed dataset
        for orig_path in replacement_originals:
            filename = os.path.basename(orig_path)
            dest_path = os.path.join(self.output_path, "mixed_dataset", filename)
            shutil.copy2(orig_path, dest_path)
        
        print(f"Replaced {replace_count} watermarked images with originals")
        print(f"Final dataset: {len(remaining_watermarked)} watermarked + {replace_count} original")
        return remaining_watermarked, replacement_originals
    
    def prepare_dataset_for_training(self):
        """Prepare the final dataset for CNN training"""
        # Get all watermarked and original images
        watermarked_dir = os.path.join(self.output_path, "mixed_dataset")
        original_dir = os.path.join(self.output_path, "original")
        
        # Collect image paths and labels
        image_paths = []
        labels = []
        
        # Add watermarked images (label = 1)
        for file in os.listdir(watermarked_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(watermarked_dir, file))
                labels.append(1)
        
        # Add original images (label = 0)
        for file in os.listdir(original_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(original_dir, file))
                labels.append(0)
        
        print(f"Total images for training: {len(image_paths)}")
        print(f"Watermarked: {sum(labels)}, Original: {len(labels) - sum(labels)}")
        
        return image_paths, labels
    
    def create_data_loaders(self, image_paths, labels, test_size=0.2, batch_size=32):
        """Create train and test data loaders - Optimized for speed"""
        
        # Define transforms with smaller image size for speed
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Reduced from 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = WatermarkDataset(X_train, y_train, transform=transform)
        test_dataset = WatermarkDataset(X_test, y_test, transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_model(self, train_loader, test_loader, epochs=10, learning_rate=0.001):
        """Train CNN model"""
        model = CNNModel().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        print(f"\nTraining model for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Evaluation phase
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device).unsqueeze(1)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            test_acc = 100 * test_correct / test_total
            
            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            # Only print every 2 epochs for speed
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Save the model
        torch.save(model.state_dict(), os.path.join(self.output_path, "watermark_detector_model.pth"))
        print(f"\nModel saved to {os.path.join(self.output_path, 'watermark_detector_model.pth')}")
        
        return model, train_losses, test_losses, train_accuracies, test_accuracies
    
    def evaluate_model(self, model, test_loader):
        """Detailed model evaluation"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                predictions = (outputs > 0.5).float().cpu().numpy()
                all_predictions.extend(predictions.flatten())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        print(f"\n=== Model Evaluation Results ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

class WatermarkDataset(Dataset):
    """Custom Dataset class for watermark detection"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class CNNModel(nn.Module):
    """CNN Model for watermark detection - Ultra minimal for instant training"""
    
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Ultra minimal convolutional layers
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)  # Reduced to 4 filters
        
        # Pooling layer
        self.pool = nn.MaxPool2d(4, 4)  # Aggressive pooling
        
        # Minimal fully connected layers
        self.fc1 = nn.Linear(4 * 32 * 32, 8)  # Very small
        self.fc2 = nn.Linear(8, 1)  # Direct to output
        
        # Minimal dropout
        self.dropout = nn.Dropout(0.1)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Single convolutional block
        x = self.pool(self.relu(self.conv1(x)))  # 128x128 -> 32x32
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Minimal fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x

def main():
    """Main function to run the complete pipeline"""
    print("=== Radioactive Watermark Detection System ===")
    
    # Initialize detector
    detector = RadioactiveWatermarkDetector()
    
    # Step 1: Load dataset
    print("\n1. Loading dataset...")
    image_files = detector.load_dataset()
    
    # Step 2: Random sampling
    print("\n2. Random sampling...")
    sampled_files, remaining_files = detector.random_sampling(image_files, sample_size=100)
    
    # Step 3: Apply watermarks
    print("\n3. Applying watermarks...")
    watermarked_files = detector.create_watermarked_dataset(sampled_files)
    
    # Step 4: Replace watermarked images
    print("\n4. Replacing watermarked images...")
    final_watermarked, final_originals = detector.replace_watermarked_images(
        watermarked_files, remaining_files, replace_count=50
    )
    
    print("\n=== Dataset Preparation Complete ===")
    print(f"Final dataset contains:")
    print(f"- Watermarked images: {len(final_watermarked)}")
    print(f"- Original images: {len(final_originals)}")
    
    # Step 5: Prepare dataset for training
    print("\n5. Preparing dataset for training...")
    image_paths, labels = detector.prepare_dataset_for_training()
    
    # Step 6: Create data loaders
    print("\n6. Creating data loaders...")
    train_loader, test_loader = detector.create_data_loaders(image_paths, labels)
    
    # Step 7: Train model
    print("\n7. Training CNN model...")
    model, train_losses, test_losses, train_accuracies, test_accuracies = detector.train_model(
        train_loader, test_loader, epochs=20
    )
    
    # Step 8: Evaluate model
    print("\n8. Evaluating model...")
    metrics = detector.evaluate_model(model, test_loader)
    
    print("\n=== Pipeline Complete ===")
    return detector, model, metrics

if __name__ == "__main__":
    main()
