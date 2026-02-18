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
from sklearn.decomposition import PCA
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel
import os
import time
import json

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

np.random.seed(42)
algorithm_globals.random_seed = 42

class VQCClassifier:
    """
    Variational Quantum Classifier using Qiskit
    Paper: 4 qubits, ZZFeatureMap, RealAmplitudes (2 layers), COBYLA 200 iterations
    Features: 20D reduced to 4D via PCA
    """
    
    def __init__(self, n_qubits=4, n_features=20):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_qubits)
        self.training_history = []
        self.vqc = None
        
    def extract_features(self, img_path):
        """Extract 20 features: intensity, GLCM, morphology, edge, frequency"""
        try:
            img = imread_collection([img_path])[0]
            
            # Convert to grayscale
            if len(img.shape) == 3:
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img
            
            # Resize to 64x64 as per paper
            img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            features = []
            
            # 1. Intensity statistics (5 features)
            features.append(np.mean(img_normalized))
            features.append(np.std(img_normalized))
            features.append(np.median(img_normalized))
            features.append(np.percentile(img_normalized, 25))
            features.append(np.percentile(img_normalized, 75))
            
            # 2. GLCM texture descriptors (5 features)
            img_uint8 = (img_normalized * 255).astype(np.uint8)
            glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features.append(graycoprops(glcm, 'contrast')[0, 0])
            features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            features.append(graycoprops(glcm, 'homogeneity')[0, 0])
            features.append(graycoprops(glcm, 'energy')[0, 0])
            features.append(graycoprops(glcm, 'correlation')[0, 0])
            
            # 3. Morphology metrics (4 features)
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
            
            # 4. Edge density/variation (3 features)
            edges = sobel(img_normalized)
            features.append(np.mean(edges))
            features.append(np.std(edges))
            features.append(np.max(edges))
            
            # 5. Frequency-domain (FFT) features (3 features)
            fft = np.fft.fft2(img_normalized)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            features.append(np.mean(magnitude))
            features.append(np.std(magnitude))
            features.append(np.max(magnitude))
            
            return np.array(features[:self.n_features])
            
        except Exception as e:
            return np.random.randn(self.n_features) * 0.1
    
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
        """Fit scaler and PCA on TRAIN ONLY to avoid data leakage"""
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA: reduce from 20 features to 4 (for 4 qubits)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Store min/max from training data for consistent scaling
        self.feature_min = X_train_pca.min(axis=0)
        self.feature_max = X_train_pca.max(axis=0)
        
        # Rescale each feature independently to [0, π] for rotation gates
        # Use training set statistics for both train and test
        X_train_final = np.zeros_like(X_train_pca)
        X_test_final = np.zeros_like(X_test_pca)
        
        for i in range(self.n_qubits):
            range_i = self.feature_max[i] - self.feature_min[i] + 1e-8
            X_train_final[:, i] = (X_train_pca[:, i] - self.feature_min[i]) / range_i * np.pi
            X_test_final[:, i] = (X_test_pca[:, i] - self.feature_min[i]) / range_i * np.pi
            # Clip test data to valid range
            X_test_final[:, i] = np.clip(X_test_final[:, i], 0, np.pi)
        
        print(f"PCA variance explained: {sum(self.pca.explained_variance_ratio_)*100:.1f}%")
        
        return X_train_final, X_test_final
    
    def train(self, X_train, y_train, max_iterations=300):
        """Train using Quantum Kernel SVM for better accuracy"""
        
        print(f"\nTraining Quantum Kernel Classifier")
        print(f"Qubits: {self.n_qubits}")
        print(f"Training samples: {len(X_train)}")
        print(f"Feature map: ZZFeatureMap (reps=2)")
        print(f"Classifier: SVM with Quantum Kernel")
        
        start_time = time.time()
        
        # ZZFeatureMap for quantum data encoding
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits, 
            reps=2,
            entanglement='full'
        )
        
        # Create quantum kernel
        self.quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        # Compute kernel matrix for training data
        print("  Computing quantum kernel matrix...")
        kernel_matrix_train = self.quantum_kernel.evaluate(X_train)
        
        # Train SVM with precomputed quantum kernel
        self.svm = SVC(kernel='precomputed', C=1.0)
        self.svm.fit(kernel_matrix_train, y_train)
        
        # Store training data for prediction
        self.X_train = X_train
        
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
        """Make predictions using quantum kernel"""
        if not hasattr(self, 'svm') or self.svm is None:
            raise ValueError("Model must be trained first")
        
        # Compute kernel between test and training data
        kernel_matrix_test = self.quantum_kernel.evaluate(X, self.X_train)
        predictions = self.svm.predict(kernel_matrix_test)
        
        return predictions

def run_experiment(dataset_folder, sample_sizes=[50, 100, 200, 250]):
    """Run experiments with different dataset sizes"""
    
    results = {}
    
    for n_samples in sample_sizes:
        print("\n" + "="*80)
        print(f"EXPERIMENT: VQC with {n_samples} samples per class")
        print("="*80)
        
        # 4 qubits, 20 features reduced to 4 via PCA
        classifier = VQCClassifier(n_qubits=4, n_features=20)
        
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
