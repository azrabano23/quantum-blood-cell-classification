#!/usr/bin/env python3
"""
Classical Convolutional Neural Network for Blood Cell Classification
====================================================================

A CNN using PyTorch for binary classification of healthy vs AML blood cells
using raw image data (not just extracted features).

Architecture:
- Conv Layer 1: 32 filters, 3x3 kernel, ReLU, MaxPool
- Conv Layer 2: 64 filters, 3x3 kernel, ReLU, MaxPool
- Conv Layer 3: 128 filters, 3x3 kernel, ReLU, MaxPool
- Flatten + Dense layers: 256 -> 128 -> 2
- Optimizer: Adam
- Loss: Cross-entropy

Author: A. Zrabano
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.io import imread_collection
from skimage.transform import resize, rotate
import os
import time
import json

np.random.seed(42)
torch.manual_seed(42)

class AugmentedDataset(Dataset):
    """Dataset with on-the-fly data augmentation"""
    def __init__(self, X, y, augment=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = torch.flip(img, [-1])
            
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = torch.flip(img, [-2])
            
            # Random rotation (±15 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                img_np = img.numpy()[0]
                img_rotated = rotate(img_np, angle, mode='reflect', preserve_range=True)
                img = torch.FloatTensor(img_rotated).unsqueeze(0)
            
            # Random brightness adjustment (±20%)
            if np.random.rand() > 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                img = torch.clamp(img * brightness_factor, 0, 1)
            
            # Random zoom (90%-110%)
            if np.random.rand() > 0.5:
                zoom_factor = np.random.uniform(0.9, 1.1)
                # Simple center crop/pad for zoom
                h, w = img.shape[-2:]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                if zoom_factor > 1.0:  # Zoom in (crop)
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    # Resize and crop
                    img_np = img.numpy()[0]
                    from skimage.transform import resize
                    img_resized = resize(img_np, (new_h, new_w), preserve_range=True, anti_aliasing=True)
                    img_cropped = img_resized[start_h:start_h+h, start_w:start_w+w]
                    img = torch.FloatTensor(img_cropped).unsqueeze(0)
        
        return img, label

class BloodCellCNN(nn.Module):
    """
    CNN for blood cell classification
    """
    def __init__(self, input_channels=1, num_classes=2):
        super(BloodCellCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128)
        )
        
        # For 64x64 images: after 3 maxpools (2x2), dimensions are 8x8
        # 128 channels * 8 * 8 = 8192
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased from 0.5
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc_layers(x)
        return x

class ClassicalCNNClassifier:
    """
    Classical CNN Classifier for blood cells
    """
    
    def __init__(self, img_size=64):
        self.img_size = img_size
        self.model = BloodCellCNN()
        self.training_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_image(self, img_path):
        """Load and preprocess image"""
        try:
            img = imread_collection([img_path])[0]
            
            # Convert to grayscale
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            
            # Resize to fixed size
            img_resized = resize(img, (self.img_size, self.img_size), anti_aliasing=True)
            
            # Normalize to [0, 1]
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            return img_normalized
            
        except Exception as e:
            # Return blank image on error
            return np.zeros((self.img_size, self.img_size))
    
    def load_data(self, dataset_folder, max_samples_per_class=150):
        """Load blood cell image data"""
        print(f"Loading image data from: {dataset_folder}")
        
        healthy_cell_types = ['LYT', 'MON', 'NGS', 'NGB']
        aml_cell_types = ['MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO', 'EOS', 'LYA', 'MYO', 'PMO']
        
        X, y = [], []
        class_counts = {'healthy': 0, 'aml': 0}
        
        for dirpath, _, filenames in os.walk(dataset_folder):
            path_parts = dirpath.split(os.sep)
            cell_type = None
            
            for part in path_parts:
                if part in healthy_cell_types or part in aml_cell_types:
                    cell_type = part
                    break
            
            if cell_type is None:
                continue
            
            for file in filenames:
                if file.endswith(('.jpg', '.png', '.tiff', '.tif')):
                    if cell_type in healthy_cell_types:
                        if class_counts['healthy'] >= max_samples_per_class:
                            continue
                        label = 0
                        class_counts['healthy'] += 1
                    elif cell_type in aml_cell_types:
                        if class_counts['aml'] >= max_samples_per_class:
                            continue
                        label = 1
                        class_counts['aml'] += 1
                    else:
                        continue
                    
                    img_path = os.path.join(dirpath, file)
                    img = self.load_image(img_path)
                    X.append(img)
                    y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        X = X[:, np.newaxis, :, :]
        
        print(f"Loaded {len(X)} images: Healthy={class_counts['healthy']}, AML={class_counts['aml']}")
        
        return X, y
    
    def train(self, X_train, y_train, epochs=60, batch_size=16, learning_rate=0.001, weight_decay=0.0001, use_augmentation=True):
        """Train the CNN with data augmentation and regularization"""
        
        print(f"\nTraining Enhanced CNN")
        print(f"Architecture: Conv(32) -> Conv(64) -> Conv(128) -> FC(256) -> FC(128) -> FC(2)")
        print(f"Training samples: {len(X_train)}")
        print(f"Image size: {self.img_size}x{self.img_size}")
        print(f"Using: Data Augmentation={use_augmentation}, Weight Decay={weight_decay}, Dropout=0.6/0.5")
        
        # Create augmented dataset
        dataset = AugmentedDataset(X_train, y_train, augment=use_augmentation)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Loss and optimizer with weight decay (L2 regularization)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Cosine annealing learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.01)
        
        start_time = time.time()
        
        self.training_history = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate accuracy on training set (without augmentation)
            with torch.no_grad():
                self.model.eval()
                X_eval_tensor = torch.FloatTensor(X_train).to(self.device)
                y_eval_tensor = torch.LongTensor(y_train).to(self.device)
                outputs = self.model(X_eval_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_eval_tensor).float().mean().item()
            
            self.training_history.append({
                'epoch': epoch,
                'loss': epoch_loss / len(dataloader),
                'accuracy': accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss/len(dataloader):.4f}, Accuracy = {accuracy:.3f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return training_time
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy(), probabilities.cpu().numpy()

def run_experiment(dataset_folder, sample_sizes=[50, 100, 200, 250]):
    """Run experiments with different dataset sizes"""
    
    results = {}
    
    for n_samples in sample_sizes:
        print("\n" + "="*80)
        print(f"EXPERIMENT: CNN with {n_samples} samples per class")
        print("="*80)
        
        classifier = ClassicalCNNClassifier(img_size=64)
        
        # Load data
        start_load = time.time()
        X, y = classifier.load_data(dataset_folder, max_samples_per_class=n_samples)
        load_time = time.time() - start_load
        
        if len(X) == 0:
            print("No data loaded. Skipping.")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Train with data augmentation and regularization (more epochs for high accuracy)
        train_time = classifier.train(X_train, y_train, epochs=150, batch_size=16, 
                                      learning_rate=0.001, weight_decay=0.0001, use_augmentation=True)
        
        # Predict
        start_pred = time.time()
        predictions, probabilities = classifier.predict(X_test)
        pred_time = time.time() - start_pred
        
        # Evaluate
        test_accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Healthy', 'AML'], output_dict=True)
        
        print(f"\nResults:")
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Load Time: {load_time:.2f}s")
        print(f"  Training Time: {train_time:.2f}s")
        print(f"  Prediction Time: {pred_time:.2f}s")
        print(f"  Total Time: {load_time + train_time + pred_time:.2f}s")
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Healthy', 'AML']))
        
        # Store results
        results[n_samples] = {
            'accuracy': float(test_accuracy),
            'load_time': float(load_time),
            'train_time': float(train_time),
            'prediction_time': float(pred_time),
            'total_time': float(load_time + train_time + pred_time),
            'precision_healthy': float(report['Healthy']['precision']),
            'recall_healthy': float(report['Healthy']['recall']),
            'f1_healthy': float(report['Healthy']['f1-score']),
            'precision_aml': float(report['AML']['precision']),
            'recall_aml': float(report['AML']['recall']),
            'f1_aml': float(report['AML']['f1-score']),
            'training_history': classifier.training_history
        }
    
    # Save results
    with open('results_cnn.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("All experiments completed. Results saved to results_cnn.json")
    print("="*80)
    
    return results

if __name__ == "__main__":
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
    else:
        results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
