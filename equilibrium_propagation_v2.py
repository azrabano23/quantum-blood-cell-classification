#!/usr/bin/env python3
"""
Equilibrium Propagation V2 - Optimized Version
===============================================

Refined version focusing on what works:
- Enhanced 20 features (stat + GLCM + morph + edge + freq)
- Deeper architecture [20, 256, 128, 64, 2]
- Simplified training (removed aggressive state normalization)
- Better hyperparameter tuning
- Longer training with patience

Author: A. Zrabano
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel
import os
import time
import json

np.random.seed(42)

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class EPNetworkV2:
    """Refined Equilibrium Propagation Network"""
    
    def __init__(self, layer_sizes=[20, 256, 128, 64, 2], beta=0.5, learning_rate=0.1, 
                 momentum=0.95, l2_reg=0.00005):
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes)
        self.momentum = momentum
        self.l2_reg = l2_reg
        
        # Initialize weights with He initialization (better for ReLU-like)
        self.weights = []
        self.biases = []
        self.weight_momentum = []
        self.bias_momentum = []
        
        for i in range(self.n_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            # He initialization
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)
            self.weight_momentum.append(np.zeros_like(w))
            self.bias_momentum.append(np.zeros_like(b))
        
        self.training_history = []
    
    def forward_pass(self, x, target=None, beta=0, n_iterations=60):
        """Simplified forward pass without aggressive normalization"""
        # Initialize states
        states = [x.copy()]
        for i in range(1, self.n_layers):
            states.append(np.ones(self.layer_sizes[i]) * 0.5)
        
        # Relax to equilibrium
        alpha = 0.4  # Relaxation rate
        
        for iteration in range(n_iterations):
            for i in range(1, self.n_layers):
                h = states[i-1] @ self.weights[i-1] + self.biases[i-1]
                
                if i == self.n_layers - 1 and target is not None:
                    h += beta * (target - states[i])
                
                new_state = sigmoid(h)
                states[i] = (1 - alpha) * states[i] + alpha * new_state
        
        return states
    
    def train_sample(self, x, y):
        """Train on single sample"""
        target = np.zeros(self.layer_sizes[-1])
        target[y] = 1.0
        
        # Two phases
        states_free = self.forward_pass(x, target=None, beta=0, n_iterations=60)
        states_nudged = self.forward_pass(x, target=target, beta=self.beta, n_iterations=60)
        
        # Update weights
        for i in range(len(self.weights)):
            # Hebbian-like gradient
            grad = (np.outer(states_nudged[i], states_nudged[i+1]) - 
                   np.outer(states_free[i], states_free[i+1])) / self.beta
            
            # L2 regularization
            grad -= self.l2_reg * self.weights[i]
            
            # Clip for stability
            grad = np.clip(grad, -0.5, 0.5)
            
            # Momentum update
            self.weight_momentum[i] = (self.momentum * self.weight_momentum[i] + 
                                      self.learning_rate * grad)
            self.weights[i] += self.weight_momentum[i]
            
            # Bias update
            bias_grad = (states_nudged[i+1] - states_free[i+1]) / self.beta
            bias_grad = np.clip(bias_grad, -0.5, 0.5)
            self.bias_momentum[i] = (self.momentum * self.bias_momentum[i] + 
                                    self.learning_rate * bias_grad)
            self.biases[i] += self.bias_momentum[i]
        
        return np.argmax(states_free[-1])
    
    def predict_sample(self, x):
        """Predict single sample"""
        states = self.forward_pass(x, target=None, beta=0, n_iterations=60)
        return np.argmax(states[-1])
    
    def train(self, X_train, y_train, epochs=150):
        """Train with simple decay schedule"""
        n_samples = len(X_train)
        initial_lr = self.learning_rate
        
        for epoch in range(epochs):
            # Simple exponential decay
            decay_factor = 0.985
            self.learning_rate = initial_lr * (decay_factor ** epoch)
            
            indices = np.random.permutation(n_samples)
            correct = 0
            
            for idx in indices:
                pred = self.train_sample(X_train[idx], y_train[idx])
                if pred == y_train[idx]:
                    correct += 1
            
            accuracy = correct / n_samples
            self.training_history.append({
                'epoch': epoch,
                'accuracy': accuracy,
                'learning_rate': self.learning_rate
            })
            
            if (epoch + 1) % 15 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train Accuracy = {accuracy:.3f}, LR = {self.learning_rate:.5f}")
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(x) for x in X])

class EPClassifierV2:
    """EP Classifier V2 for blood cells"""
    
    def __init__(self, layer_sizes=[20, 256, 128, 64, 2]):
        self.layer_sizes = layer_sizes
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = []
        
    def extract_features(self, img_path):
        """Extract 20 enhanced features"""
        try:
            img = imread_collection([img_path])[0]
            
            if len(img.shape) == 3:
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img
            
            img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            features = []
            
            # 1. Statistical features (6)
            features.append(np.mean(img_normalized))
            features.append(np.std(img_normalized))
            features.append(np.median(img_normalized))
            features.append(np.percentile(img_normalized, 25))
            features.append(np.percentile(img_normalized, 75))
            features.append(np.max(img_normalized) - np.min(img_normalized))
            
            # 2. GLCM texture features (6)
            img_uint8 = (img_normalized * 255).astype(np.uint8)
            glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features.append(graycoprops(glcm, 'contrast')[0, 0])
            features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            features.append(graycoprops(glcm, 'homogeneity')[0, 0])
            features.append(graycoprops(glcm, 'energy')[0, 0])
            features.append(graycoprops(glcm, 'correlation')[0, 0])
            features.append(graycoprops(glcm, 'ASM')[0, 0])
            
            # 3. Morphological features (4)
            thresh = img_normalized > np.mean(img_normalized)
            labeled = label(thresh)
            if labeled.max() > 0:
                props = regionprops(labeled)[0]
                features.append(props.area / (64 * 64))
                features.append(props.eccentricity)
                features.append(props.solidity)
                features.append(props.extent)
            else:
                features.extend([0.5, 0.5, 0.5, 0.5])
            
            # 4. Edge features (2)
            edges = sobel(img_normalized)
            features.append(np.mean(edges))
            features.append(np.std(edges))
            
            # 5. Frequency domain features (2)
            fft = np.fft.fft2(img_normalized)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            features.append(np.mean(magnitude_spectrum))
            features.append(np.std(magnitude_spectrum))
            
            return np.array(features[:self.layer_sizes[0]])
            
        except Exception as e:
            return np.random.randn(self.layer_sizes[0]) * 0.1
    
    def load_data(self, dataset_folder, max_samples_per_class=150):
        """Load blood cell data"""
        print(f"Loading data from: {dataset_folder}")
        
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
                    features = self.extract_features(img_path)
                    X.append(features)
                    y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        X = self.scaler.fit_transform(X)
        
        print(f"Loaded {len(X)} samples: Healthy={class_counts['healthy']}, AML={class_counts['aml']}")
        
        return X, y
    
    def train(self, X_train, y_train, epochs=150):
        """Train EP network V2"""
        print(f"\nTraining EP Network V2")
        print(f"Architecture: {' -> '.join(map(str, self.layer_sizes))}")
        print(f"Training samples: {len(X_train)}")
        print(f"Hyperparams: beta=0.5, lr=0.1->decay, momentum=0.95, l2=0.00005")
        
        start_time = time.time()
        
        self.model = EPNetworkV2(
            layer_sizes=self.layer_sizes,
            beta=0.5,
            learning_rate=0.1,
            momentum=0.95,
            l2_reg=0.00005
        )
        
        self.model.train(X_train, y_train, epochs=epochs)
        self.training_history = self.model.training_history
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return training_time
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)

def run_experiment(dataset_folder, sample_sizes=[50, 100, 200, 250]):
    """Run experiments with different dataset sizes"""
    results = {}
    
    for n_samples in sample_sizes:
        print("\n" + "="*80)
        print(f"EXPERIMENT: EP V2 with {n_samples} samples per class")
        print("="*80)
        
        classifier = EPClassifierV2(layer_sizes=[20, 256, 128, 64, 2])
        
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
        
        # Train
        train_time = classifier.train(X_train, y_train, epochs=150)
        
        # Predict
        start_pred = time.time()
        predictions = classifier.predict(X_test)
        pred_time = time.time() - start_pred
        
        # Evaluate
        test_accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Healthy', 'AML'], output_dict=True, zero_division=0)
        
        print(f"\nResults:")
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Load Time: {load_time:.2f}s")
        print(f"  Training Time: {train_time:.2f}s")
        print(f"  Prediction Time: {pred_time:.2f}s")
        print(f"  Total Time: {load_time + train_time + pred_time:.2f}s")
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Healthy', 'AML'], zero_division=0))
        
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
    with open('results_ep_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("All experiments completed. Results saved to results_ep_v2.json")
    print("="*80)
    
    return results

if __name__ == "__main__":
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
    else:
        results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
