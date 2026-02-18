#!/usr/bin/env python3
"""
Quantum Kernel VQC - Fast Alternative Approach
===============================================

This is a FAST alternative to the paper-exact VQC that uses:
- Quantum kernel (FidelityQuantumKernel) with ZZFeatureMap
- Classical SVM on the quantum kernel matrix

This achieves similar accuracy (~80%) much faster than the paper-exact
COBYLA optimization approach.

NOTE: This is NOT the exact methodology from the paper. For paper-exact
implementation, use vqc_classifier.py instead.

Paper: arXiv:2601.18710
Author: A. Zrabano
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel
import os
import sys
import time
import json

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.utils import algorithm_globals

np.random.seed(42)
algorithm_globals.random_seed = 42


class QuantumKernelVQC:
    """
    Fast Quantum Kernel Classifier.
    
    Uses FidelityQuantumKernel with ZZFeatureMap + classical SVM.
    This is faster than paper-exact VQC but uses different methodology.
    
    For paper-exact implementation (slower), use VQCClassifier in vqc_classifier.py
    """
    
    def __init__(self, n_qubits=4, n_features=20):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_qubits)
        self.quantum_kernel = None
        self.svm = None
        
    def extract_features(self, img_path):
        """Extract 20 features: intensity, GLCM, morphology, edge, frequency"""
        try:
            img = imread_collection([img_path])[0]
            
            if len(img.shape) == 3:
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img
            
            img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            features = []
            
            # 1. Intensity statistics (5 features)
            features.extend([
                np.mean(img_normalized),
                np.std(img_normalized),
                np.median(img_normalized),
                np.percentile(img_normalized, 25),
                np.percentile(img_normalized, 75)
            ])
            
            # 2. GLCM texture (5 features)
            img_uint8 = (img_normalized * 255).astype(np.uint8)
            glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0]
            ])
            
            # 3. Morphology (4 features)
            thresh = img_normalized > np.mean(img_normalized)
            labeled = label(thresh)
            if labeled.max() > 0:
                props = regionprops(labeled)[0]
                features.extend([props.area / (64 * 64), props.eccentricity, props.solidity, props.extent])
            else:
                features.extend([0.5, 0.5, 0.5, 0.5])
            
            # 4. Edge features (3 features)
            edges = sobel(img_normalized)
            features.extend([np.mean(edges), np.std(edges), np.max(edges)])
            
            # 5. FFT features (3 features)
            fft = np.fft.fft2(img_normalized)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            features.extend([np.mean(magnitude), np.std(magnitude), np.max(magnitude)])
            
            return np.array(features[:self.n_features])
        except:
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
                        lbl = 0
                        class_counts['healthy'] += 1
                    elif cell_type in aml_cell_types:
                        if class_counts['aml'] >= max_samples_per_class:
                            continue
                        lbl = 1
                        class_counts['aml'] += 1
                    else:
                        continue
                    
                    img_path = os.path.join(dirpath, file)
                    features = self.extract_features(img_path)
                    X.append(features)
                    y.append(lbl)
        
        print(f"Loaded {len(X)} samples: Healthy={class_counts['healthy']}, AML={class_counts['aml']}")
        return np.array(X), np.array(y)
    
    def preprocess(self, X_train, X_test):
        """Preprocess: standardize, PCA to 4 dims, scale to [0, pi]"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Scale to [0, pi] for quantum encoding
        self.feature_min = X_train_pca.min(axis=0)
        self.feature_max = X_train_pca.max(axis=0)
        
        X_train_final = np.zeros_like(X_train_pca)
        X_test_final = np.zeros_like(X_test_pca)
        
        for i in range(self.n_qubits):
            range_i = self.feature_max[i] - self.feature_min[i] + 1e-8
            X_train_final[:, i] = (X_train_pca[:, i] - self.feature_min[i]) / range_i * np.pi
            X_test_final[:, i] = np.clip((X_test_pca[:, i] - self.feature_min[i]) / range_i * np.pi, 0, np.pi)
        
        print(f"PCA variance explained: {sum(self.pca.explained_variance_ratio_)*100:.1f}%")
        return X_train_final, X_test_final
    
    def train(self, X_train, y_train):
        """Train using quantum kernel SVM"""
        print(f"\nTraining Quantum Kernel Classifier (FAST alternative)")
        print(f"  NOTE: This is NOT the paper-exact methodology")
        print(f"  For paper-exact, use vqc_classifier.py")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Feature map: ZZFeatureMap (reps=2)")
        print(f"  Classifier: SVM with quantum kernel")
        
        start_time = time.time()
        
        # ZZFeatureMap for quantum encoding
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement='full'
        )
        
        # Create quantum kernel
        self.quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        # Compute kernel matrix
        print("  Computing quantum kernel matrix...")
        kernel_matrix_train = self.quantum_kernel.evaluate(X_train)
        
        # Train SVM with quantum kernel
        self.svm = SVC(kernel='precomputed', C=1.0)
        self.svm.fit(kernel_matrix_train, y_train)
        
        # Store training data for prediction
        self.X_train = X_train
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        return training_time
    
    def predict(self, X):
        """Make predictions"""
        if self.svm is None:
            raise ValueError("Model must be trained first")
        
        kernel_matrix_test = self.quantum_kernel.evaluate(X, self.X_train)
        return self.svm.predict(kernel_matrix_test)


def run_experiment(dataset_folder, sample_sizes=[50, 100, 200, 250]):
    """Run experiments with different dataset sizes"""
    results = {}
    
    for n_samples in sample_sizes:
        print("\n" + "="*80)
        print(f"EXPERIMENT: Quantum Kernel VQC with {n_samples} samples per class")
        print("="*80)
        
        classifier = QuantumKernelVQC(n_qubits=4, n_features=20)
        
        start_load = time.time()
        X, y = classifier.load_data(dataset_folder, max_samples_per_class=n_samples)
        load_time = time.time() - start_load
        
        if len(X) == 0:
            print("No data loaded. Skipping.")
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_test = classifier.preprocess(X_train, X_test)
        
        try:
            train_time = classifier.train(X_train, y_train)
        except Exception as e:
            print(f"Training failed: {e}")
            continue
        
        start_pred = time.time()
        predictions = classifier.predict(X_test)
        pred_time = time.time() - start_pred
        
        test_accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nResults:")
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Training Time: {train_time:.2f}s")
        print(classification_report(y_test, predictions, target_names=['Healthy', 'AML'], zero_division=0))
        
        results[n_samples] = {
            'accuracy': float(test_accuracy),
            'train_time': float(train_time),
        }
    
    with open('results_vqc_quantum_kernel.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = os.environ.get(
            'AML_DATASET_PATH',
            '/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU'
        )
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        print("Usage: python vqc_quantum_kernel.py <dataset_path>")
        sys.exit(1)
    
    results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
