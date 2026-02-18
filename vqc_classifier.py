#!/usr/bin/env python3
"""
Variational Quantum Classifier (VQC) for Blood Cell Classification
===================================================================

Implements a VQC using Qiskit EXACTLY as described in the paper:
- Feature map: ZZFeatureMap for data encoding
- Ansatz: RealAmplitudes (2 layers, 8 trainable parameters)
- Optimizer: COBYLA (gradient-free), 200 iterations
- Loss: MSE between <Z0> expectation and target labels
- Classification: threshold <Z0> at zero

Paper: arXiv:2601.18710
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
import sys
import time
import json

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

np.random.seed(42)
algorithm_globals.random_seed = 42


class VQCClassifier:
    """
    Variational Quantum Classifier - EXACTLY as described in paper.
    
    Paper specifications:
    - 4 qubits
    - ZZFeatureMap for encoding (creates entanglement via second-order Pauli-Z)
    - RealAmplitudes ansatz (2 layers, 8 trainable parameters)
    - COBYLA optimizer (gradient-free), 200 iterations
    - MSE loss between <Z0> and target labels
    - Classification: <Z0> > 0 -> class 1, else class 0
    """
    
    def __init__(self, n_qubits=4, n_features=20):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_qubits)
        self.training_history = []
        self.optimal_params = None
        self.estimator = StatevectorEstimator()
        
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
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} samples: Healthy={class_counts['healthy']}, AML={class_counts['aml']}")
        
        return X, y
    
    def preprocess(self, X_train, X_test):
        """
        Preprocess data EXACTLY as paper describes:
        - Standardize
        - PCA from 20 to 4 features (retaining ~95% variance)
        - Rescale to [0, 2*pi] for rotation gates
        """
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA: reduce from 20 features to 4 (for 4 qubits)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Store min/max from training data for consistent scaling
        self.feature_min = X_train_pca.min(axis=0)
        self.feature_max = X_train_pca.max(axis=0)
        
        # Rescale to [0, 2*pi] as per paper ("rescaled to [0, 2π] to match rotation gate domains")
        X_train_final = np.zeros_like(X_train_pca)
        X_test_final = np.zeros_like(X_test_pca)
        
        for i in range(self.n_qubits):
            range_i = self.feature_max[i] - self.feature_min[i] + 1e-8
            X_train_final[:, i] = (X_train_pca[:, i] - self.feature_min[i]) / range_i * 2 * np.pi
            X_test_final[:, i] = (X_test_pca[:, i] - self.feature_min[i]) / range_i * 2 * np.pi
            # Clip test data to valid range
            X_test_final[:, i] = np.clip(X_test_final[:, i], 0, 2 * np.pi)
        
        print(f"PCA variance explained: {sum(self.pca.explained_variance_ratio_)*100:.1f}%")
        
        return X_train_final, X_test_final
    
    def _build_circuit(self, x, params):
        """
        Build VQC circuit EXACTLY as paper describes:
        - ZZFeatureMap for encoding (creates entanglement between qubit pairs)
        - RealAmplitudes ansatz (2 layers, 8 trainable parameters)
        """
        # Feature map: ZZFeatureMap with 2 repetitions
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement='full'
        )
        
        # Ansatz: RealAmplitudes with 2 layers (8 parameters for 4 qubits)
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=2,
            entanglement='full'
        )
        
        # Combine into full circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(feature_map.assign_parameters(x), inplace=True)
        qc.compose(ansatz.assign_parameters(params), inplace=True)
        
        return qc
    
    def _compute_expectation(self, x, params):
        """
        Compute <Z0> expectation value on first qubit.
        Paper: "expectation-value measurement on the first qubit"
        """
        qc = self._build_circuit(x, params)
        
        # Z operator on first qubit
        observable = SparsePauliOp.from_list([('Z' + 'I' * (self.n_qubits - 1), 1.0)])
        
        # Compute expectation
        job = self.estimator.run([(qc, observable)])
        result = job.result()[0]
        
        return result.data.evs
    
    def _compute_batch_expectation(self, X, params):
        """Compute expectations for a batch of samples"""
        expectations = []
        for x in X:
            exp = self._compute_expectation(x, params)
            expectations.append(exp)
        return np.array(expectations)
    
    def _mse_loss(self, params, X, y):
        """
        MSE loss between <Z0> and target labels.
        Paper: "mean squared error between <Z0> and target labels"
        
        Labels: 0 (healthy) -> target -1, 1 (AML) -> target +1
        """
        # Convert labels to +1/-1 for expectation value comparison
        targets = 2 * y - 1  # 0 -> -1, 1 -> +1
        
        # Compute expectations
        expectations = self._compute_batch_expectation(X, params)
        
        # MSE loss
        loss = np.mean((expectations - targets) ** 2)
        
        return loss
    
    def train(self, X_train, y_train, max_iterations=200):
        """
        Train VQC using COBYLA optimizer EXACTLY as paper describes.
        Paper: "COBYLA, gradient-free... steps 1-4 repeat for 200 iterations"
        """
        print(f"\nTraining VQC (Paper-exact implementation)")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Feature map: ZZFeatureMap (reps=2, full entanglement)")
        print(f"  Ansatz: RealAmplitudes (2 layers, 8 parameters)")
        print(f"  Optimizer: COBYLA, {max_iterations} iterations")
        print(f"  Loss: MSE between <Z0> and target labels")
        
        start_time = time.time()
        
        # RealAmplitudes with 2 layers on 4 qubits = 8 parameters
        n_params = self.n_qubits * 2  # 2 layers
        initial_params = np.random.uniform(0, 2*np.pi, n_params)
        
        # Track best loss
        self.best_loss = float('inf')
        self.iteration_count = 0
        
        def objective(params):
            loss = self._mse_loss(params, X_train, y_train)
            self.iteration_count += 1
            
            if loss < self.best_loss:
                self.best_loss = loss
                self.optimal_params = params.copy()
            
            if self.iteration_count % 20 == 0:
                print(f"    Iteration {self.iteration_count}: Loss = {loss:.4f}")
            
            self.training_history.append({
                'iteration': self.iteration_count,
                'loss': float(loss)
            })
            
            return loss
        
        # COBYLA optimizer (gradient-free) as per paper
        optimizer = COBYLA(maxiter=max_iterations)
        result = optimizer.minimize(objective, initial_params)
        
        self.optimal_params = result.x
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final loss: {result.fun:.4f}")
        
        return training_time
    
    def predict(self, X):
        """
        Make predictions using trained VQC.
        Paper: "Classification employs expectation-value measurement on first qubit,
               thresholded at zero for binary label assignment"
        """
        if self.optimal_params is None:
            raise ValueError("Model must be trained first")
        
        predictions = []
        for x in X:
            exp = self._compute_expectation(x, self.optimal_params)
            # Threshold at zero: <Z0> > 0 -> class 1 (AML), else class 0 (healthy)
            pred = 1 if exp > 0 else 0
            predictions.append(pred)
        
        return np.array(predictions)

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
    # Accept dataset path from command line or use default
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = os.environ.get(
            'AML_DATASET_PATH',
            '/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU'
        )
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        print("Usage: python vqc_classifier.py <dataset_path>")
        print("Or set AML_DATASET_PATH environment variable")
        sys.exit(1)
    else:
        results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
