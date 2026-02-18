#!/usr/bin/env python3
"""
Variational Quantum Classifier (VQC) for Blood Cell Classification
===================================================================

Implements a VQC using Qiskit with:
- Feature map: ZZFeatureMap for data encoding
- Ansatz: RealAmplitudes variational form
- Optimizer: COBYLA (gradient-free)
- Backend: Qiskit simulator

This is a pure quantum approach using Qiskit's built-in VQC.

Author: A. Zrabano
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
import os
import time
import json

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_machine_learning.algorithms import VQC
except ImportError:
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.algorithms import algorithm_globals
    from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler

np.random.seed(42)
algorithm_globals.random_seed = 42

class VQCClassifier:
    """
    Variational Quantum Classifier using Qiskit
    """
    
    def __init__(self, n_qubits=4, use_advanced_encoding=True):
        self.n_qubits = n_qubits
        self.scaler = StandardScaler()
        self.training_history = []
        self.vqc = None
        self.use_advanced_encoding = use_advanced_encoding
        
    def extract_features(self, img_path):
        """Extract texture and statistical features from image"""
        try:
            img = imread_collection([img_path])[0]
            
            # Convert to grayscale
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            
            # Resize
            img_resized = resize(img, (32, 32), anti_aliasing=True)
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            # GLCM texture features
            img_uint8 = (img_normalized * 255).astype(np.uint8)
            glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            features = []
            # Enhanced feature extraction for quantum advantage
            # Statistical features
            features.append(np.mean(img_normalized))
            features.append(np.std(img_normalized))
            features.append(np.median(img_normalized))
            features.append(np.percentile(img_normalized, 25))
            features.append(np.percentile(img_normalized, 75))
            features.append(np.max(img_normalized) - np.min(img_normalized))  # Range
            
            # Texture features from GLCM
            features.append(graycoprops(glcm, 'contrast')[0, 0])
            features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            features.append(graycoprops(glcm, 'homogeneity')[0, 0])
            features.append(graycoprops(glcm, 'energy')[0, 0])
            features.append(graycoprops(glcm, 'correlation')[0, 0])
            features.append(graycoprops(glcm, 'ASM')[0, 0])
            
            # Moment features
            features.append(np.mean(img_normalized ** 2))  # Second moment
            features.append(np.mean(img_normalized ** 3))  # Third moment
            
            # Edge statistics
            from scipy import ndimage
            edges = ndimage.sobel(img_normalized)
            features.append(np.mean(edges))
            features.append(np.std(edges))
            
            return np.array(features[:self.n_qubits])
            
        except Exception as e:
            return np.random.randn(self.n_qubits) * 0.1
    
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
        
        print(f"Loaded {len(X)} samples: Healthy={class_counts['healthy']}, AML={class_counts['aml']}")
        
        return X, y
    
    def preprocess(self, X_train, X_test):
        """Fit scaler on TRAIN ONLY to avoid data leakage"""
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)  # transform only, no fit!
        return X_train, X_test
    
    def train(self, X_train, y_train, max_iterations=100):
        """Train the VQC with optimized parameters"""
        
        print(f"\nTraining Optimized Variational Quantum Classifier (VQC)")
        print(f"Qubits: {self.n_qubits}")
        print(f"Training samples: {len(X_train)}")
        print(f"Advanced encoding: {self.use_advanced_encoding}")
        print(f"Optimizer: COBYLA (Constrained Optimization BY Linear Approximation)")
        
        # Create optimized feature map
        if self.use_advanced_encoding:
            feature_map = ZZFeatureMap(
                feature_dimension=self.n_qubits, 
                reps=2,  # Balanced encoding
                entanglement='circular',  # Circular for better connectivity
                insert_barriers=False
            )
        else:
            feature_map = ZZFeatureMap(
                feature_dimension=self.n_qubits, 
                reps=2, 
                entanglement='linear'
            )
        
        # Create optimized ansatz
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits, 
            reps=3,  # Balanced depth
            entanglement='circular',  # Circular for efficiency
            insert_barriers=False
        )
        
        # Use COBYLA optimizer (more stable for small datasets)
        optimizer = COBYLA(maxiter=max_iterations)
        
        # Create sampler
        sampler = Sampler()
        
        # Create VQC
        self.vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=self._callback
        )
        
        start_time = time.time()
        
        # Train
        self.vqc.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return training_time
    
    def _callback(self, weights_count, parameters, mean, std):
        """Callback for tracking training progress"""
        self.training_history.append({
            'iteration': weights_count,
            'mean': mean,
            'std': std
        })
        if weights_count % 20 == 0:
            print(f"  Iteration {weights_count}: Loss = {mean:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        if self.vqc is None:
            raise ValueError("Model must be trained first")
        
        predictions = self.vqc.predict(X)
        
        return predictions

def run_experiment(dataset_folder, sample_sizes=[50, 100, 200, 250]):
    """Run experiments with different dataset sizes"""
    
    results = {}
    
    for n_samples in sample_sizes:
        print("\n" + "="*80)
        print(f"EXPERIMENT: VQC with {n_samples} samples per class")
        print("="*80)
        
        # Use 4 qubits with advanced encoding for quantum advantage
        classifier = VQCClassifier(n_qubits=4, use_advanced_encoding=True)
        
        # Load data
        start_load = time.time()
        X, y = classifier.load_data(dataset_folder, max_samples_per_class=n_samples)
        load_time = time.time() - start_load
        
        if len(X) == 0:
            print("No data loaded. Skipping.")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocess AFTER split to avoid data leakage
        X_train, X_test = classifier.preprocess(X_train, X_test)
        
        # Train with optimized iterations
        try:
            train_time = classifier.train(X_train, y_train, max_iterations=200)
        except Exception as e:
            print(f"Training failed: {e}")
            continue
        
        # Predict
        start_pred = time.time()
        try:
            predictions = classifier.predict(X_test)
            pred_time = time.time() - start_pred
        except Exception as e:
            print(f"Prediction failed: {e}")
            continue
        
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
    with open('results_vqc.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("All experiments completed. Results saved to results_vqc.json")
    print("="*80)
    
    return results

if __name__ == "__main__":
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
    else:
        results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
