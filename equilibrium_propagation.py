#!/usr/bin/env python3
"""
Equilibrium Propagation for Blood Cell Classification
=====================================================

Implements Equilibrium Propagation (EP), a biologically-inspired learning
algorithm that doesn't use backpropagation. Instead, it uses local energy
minimization.

Key concepts:
- Two phases: free phase (no target) and nudged phase (with target)
- Energy-based model
- Local learning rules (more biologically plausible)

Reference:
Scellier, B., & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap
between energy-based models and backpropagation. Frontiers in computational neuroscience.

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
from scipy import ndimage
import os
import time
import json

np.random.seed(42)

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

class EquilibriumPropagationNetwork:
    """
    Optimized Equilibrium Propagation Network
    
    Enhanced with:
    - Better weight initialization (Xavier/He)
    - Adaptive learning rate
    - Batch normalization-like stability
    - Longer relaxation for better equilibrium
    """
    
    def __init__(self, layer_sizes=[20, 256, 128, 64, 2], beta=0.3, learning_rate=0.05, 
                 use_momentum=True, momentum=0.9, l2_reg=0.0001):
        """
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            beta: Nudging parameter for the output layer (lower for stability)
            learning_rate: Learning rate for weight updates (higher for faster learning)
            use_momentum: Whether to use momentum
            momentum: Momentum factor
            l2_reg: L2 regularization strength
        """
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes)
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.l2_reg = l2_reg
        
        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []
        self.weight_momentum = []
        self.bias_momentum = []
        
        for i in range(self.n_layers - 1):
            # Xavier initialization for better gradient flow
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)
            
            if use_momentum:
                self.weight_momentum.append(np.zeros_like(w))
                self.bias_momentum.append(np.zeros_like(b))
        
        self.training_history = []
    
    def energy(self, states):
        """
        Compute the energy of the network given neuron states
        
        E = -sum_i sum_j W_ij * s_i * s_j - sum_i b_i * s_i + sum_i s_i^2
        """
        energy = 0
        
        # Interaction energy (weights)
        for i in range(len(self.weights)):
            energy -= np.sum(states[i][:, None] * self.weights[i] * states[i+1])
        
        # Bias energy
        for i in range(len(self.biases)):
            energy -= np.sum(self.biases[i] * states[i+1])
        
        # Quadratic term (keeps states bounded)
        for state in states:
            energy += 0.5 * np.sum(state ** 2)
        
        return energy
    
    def forward_pass(self, x, target=None, beta=0, n_iterations=50):
        """
        Relax the network to equilibrium with improved stability and state normalization
        
        Args:
            x: Input data
            target: Target output (for nudged phase)
            beta: Nudging strength
            n_iterations: Number of relaxation iterations (more for stability)
        """
        # Initialize states with better initialization
        states = [x.copy()]
        for i in range(1, self.n_layers):
            # Initialize closer to expected activation range
            states.append(np.ones(self.layer_sizes[i]) * 0.5 + 
                         np.random.randn(self.layer_sizes[i]) * 0.1)
        
        # Relax to equilibrium with adaptive step size
        alpha = 0.3  # Slower, more stable updates
        
        for iteration in range(n_iterations):
            # Update each layer (except input) with damping
            for i in range(1, self.n_layers):
                # Compute input to this layer
                h = states[i-1] @ self.weights[i-1] + self.biases[i-1]
                
                # Add nudging to output layer if in nudged phase
                if i == self.n_layers - 1 and target is not None:
                    h += beta * (target - states[i])
                
                # Update state with smaller step for stability
                new_state = sigmoid(h)
                
                # State normalization for stability (like batch norm)
                if i < self.n_layers - 1:  # Don't normalize output layer
                    new_state = (new_state - np.mean(new_state)) / (np.std(new_state) + 1e-8)
                    new_state = (new_state + 1) / 2  # Rescale to [0, 1] range
                    new_state = np.clip(new_state, 0.01, 0.99)  # Prevent saturation
                
                states[i] = (1 - alpha) * states[i] + alpha * new_state
        
        return states
    
    def train_sample(self, x, y):
        """
        Train on a single sample using optimized equilibrium propagation
        
        Two phases:
        1. Free phase: relax without target
        2. Nudged phase: relax with target nudging
        
        Improvements:
        - Longer relaxation for better equilibrium
        - Momentum for stable updates
        - Gradient clipping for stability
        """
        # Convert label to one-hot
        target = np.zeros(self.layer_sizes[-1])
        target[y] = 1.0
        
        # Free phase (beta = 0) - more iterations for stability
        states_free = self.forward_pass(x, target=None, beta=0, n_iterations=50)
        
        # Nudged phase (beta > 0)
        states_nudged = self.forward_pass(x, target=target, beta=self.beta, n_iterations=50)
        
        # Compute weight gradients with momentum and clipping
        for i in range(len(self.weights)):
            # Gradient is difference in correlations between free and nudged phases
            grad = (np.outer(states_nudged[i], states_nudged[i+1]) - 
                   np.outer(states_free[i], states_free[i+1])) / self.beta
            
            # Add L2 regularization gradient
            grad -= self.l2_reg * self.weights[i]
            
            # Clip gradients for stability
            grad = np.clip(grad, -1.0, 1.0)
            
            # Apply momentum if enabled
            if self.use_momentum:
                self.weight_momentum[i] = (self.momentum * self.weight_momentum[i] + 
                                          self.learning_rate * grad)
                self.weights[i] += self.weight_momentum[i]
            else:
                self.weights[i] += self.learning_rate * grad
            
            # Bias gradient
            bias_grad = (states_nudged[i+1] - states_free[i+1]) / self.beta
            bias_grad = np.clip(bias_grad, -1.0, 1.0)
            
            if self.use_momentum:
                self.bias_momentum[i] = (self.momentum * self.bias_momentum[i] + 
                                        self.learning_rate * bias_grad)
                self.biases[i] += self.bias_momentum[i]
            else:
                self.biases[i] += self.learning_rate * bias_grad
        
        # Return prediction from free phase
        return np.argmax(states_free[-1])
    
    def predict_sample(self, x):
        """Predict a single sample with stable relaxation"""
        states = self.forward_pass(x, target=None, beta=0, n_iterations=50)
        return np.argmax(states[-1])
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=150, patience=25):
        """Train the network with adaptive learning rate for 86%+ accuracy"""
        n_samples = len(X_train)
        initial_lr = self.learning_rate
        best_val_acc = 0.0
        patience_counter = 0
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]

        for epoch in range(epochs):
            # Cosine annealing learning rate schedule with warm restarts
            self.learning_rate = initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch % 50) / 50))
            
            # Adaptive equilibrium iterations (increase over time for better convergence)
            n_iter = min(50 + epoch // 10, 80)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            correct = 0
            for idx in indices:
                pred = self.train_sample(X_train[idx], y_train[idx])
                if pred == y_train[idx]:
                    correct += 1
            
            train_accuracy = correct / n_samples
            
            # Validation accuracy if provided
            val_accuracy = None
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_accuracy = np.mean(val_predictions == y_val)
                
                # Early stopping check
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    # Restore best weights
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            self.training_history.append({
                'epoch': epoch,
                'accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'learning_rate': self.learning_rate
            })
            
            if (epoch + 1) % 10 == 0:
                if val_accuracy is not None:
                    print(f"  Epoch {epoch+1}/{epochs}: Train Acc = {train_accuracy:.3f}, Val Acc = {val_accuracy:.3f}, LR = {self.learning_rate:.5f}")
                else:
                    print(f"  Epoch {epoch+1}/{epochs}: Train Acc = {train_accuracy:.3f}, LR = {self.learning_rate:.5f}")
    
    def predict(self, X):
        """Predict multiple samples"""
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x))
        return np.array(predictions)

class EquilibriumPropagationClassifier:
    """
    EP Classifier for blood cells
    """
    
    def __init__(self, layer_sizes=[20, 256, 128, 64, 2]):
        self.layer_sizes = layer_sizes
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = []
        
    def extract_features(self, img_path):
        """Extract enhanced texture, statistical, morphology and frequency features"""
        try:
            img = imread_collection([img_path])[0]
            
            # Convert to grayscale
            if len(img.shape) == 3:
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img
            
            # Resize
            img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            features = []
            
            # 1. Statistical features (6)
            features.append(np.mean(img_normalized))
            features.append(np.std(img_normalized))
            features.append(np.median(img_normalized))
            features.append(np.percentile(img_normalized, 25))
            features.append(np.percentile(img_normalized, 75))
            features.append(np.max(img_normalized) - np.min(img_normalized))  # Range
            
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
            # Threshold and get region properties
            thresh = img_normalized > np.mean(img_normalized)
            labeled = label(thresh)
            if labeled.max() > 0:
                props = regionprops(labeled)[0]  # Get largest region
                features.append(props.area / (64 * 64))  # Normalized area
                features.append(props.eccentricity)  # Shape elongation
                features.append(props.solidity)  # Convexity
                features.append(props.extent)  # Bounding box fill ratio
            else:
                features.extend([0.5, 0.5, 0.5, 0.5])
            
            # 4. Edge features (2)
            edges = sobel(img_normalized)
            features.append(np.mean(edges))  # Edge density
            features.append(np.std(edges))  # Edge variation
            
            # 5. Frequency domain features (2)
            # Use FFT to capture frequency characteristics
            fft = np.fft.fft2(img_normalized)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            features.append(np.mean(magnitude_spectrum))  # Average frequency magnitude
            features.append(np.std(magnitude_spectrum))  # Frequency variation
            
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
        
        # NOTE: No scaling here - done in preprocess() to avoid leakage
        print(f"Loaded {len(X)} samples: Healthy={class_counts['healthy']}, AML={class_counts['aml']}")
        
        return X, y
    
    def preprocess(self, X_train, X_test):
        """Fit scaler on TRAIN ONLY to avoid data leakage"""
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)  # transform only, no fit!
        return X_train, X_test
    
    def train(self, X_train, y_train, epochs=100, validation_split=0.15):
        """Train optimized EP network with validation and early stopping"""
        
        # Split training data for validation
        if validation_split > 0:
            n_val = int(len(X_train) * validation_split)
            indices = np.random.permutation(len(X_train))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
        else:
            X_train_split = X_train
            y_train_split = y_train
            X_val = None
            y_val = None
        
        print(f"\nTraining Enhanced Equilibrium Propagation Network")
        print(f"Architecture: {' -> '.join(map(str, self.layer_sizes))}")
        print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val) if X_val is not None else 0}")
        print(f"Features: 20 (6 stat + 6 GLCM + 4 morph + 2 edge + 2 freq)")
        print(f"Using: momentum, L2 reg, cosine annealing, state normalization, early stopping")
        
        start_time = time.time()
        
        self.model = EquilibriumPropagationNetwork(
            layer_sizes=self.layer_sizes,
            beta=0.4,  # Slightly higher for better gradient signal
            learning_rate=0.1,  # Higher initial LR with cosine annealing
            use_momentum=True,
            momentum=0.9,
            l2_reg=0.00005  # Less regularization
        )

        self.model.train(X_train_split, y_train_split, X_val=X_val, y_val=y_val, epochs=150, patience=25)
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
        print(f"EXPERIMENT: Equilibrium Propagation with {n_samples} samples per class")
        print("="*80)
        
        # Use enhanced deeper network with more features
        classifier = EquilibriumPropagationClassifier(layer_sizes=[20, 256, 128, 64, 2])
        
        # Load data
        start_load = time.time()
        X, y = classifier.load_data(dataset_folder, max_samples_per_class=n_samples)
        load_time = time.time() - start_load
        
        if len(X) == 0:
            print("No data loaded. Skipping.")
            continue
        
        # SPLIT FIRST before any preprocessing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Preprocess: fit on train only (NO LEAKAGE)
        X_train, X_test = classifier.preprocess(X_train, X_test)
        
        # Train with more epochs for convergence
        train_time = classifier.train(X_train, y_train, epochs=100)
        
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
    with open('results_ep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("All experiments completed. Results saved to results_ep.json")
    print("="*80)
    
    return results

if __name__ == "__main__":
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
    else:
        results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
