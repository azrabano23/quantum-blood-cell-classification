#!/usr/bin/env python3
"""
Run VQC on IBM Quantum Hardware
================================

This script runs the blood cell classifier on real IBM Quantum computers.

Setup (one-time):
1. Go to https://quantum.ibm.com/
2. Log in and click your profile icon → "Account settings"
3. Copy your API token
4. Run: python run_on_ibm_quantum.py --setup

Author: A. Zrabano
"""

import argparse
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
import time
import json

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals

# IBM Quantum Runtime (new platform)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

np.random.seed(42)
algorithm_globals.random_seed = 42


def setup_ibm_quantum():
    """One-time setup to save IBM Quantum credentials"""
    print("\n" + "="*60)
    print("IBM Quantum Platform Setup")
    print("="*60)
    print("\n1. Go to: https://quantum.ibm.com/")
    print("2. Log in with your IBM Cloud account")
    print("3. From the dashboard, click 'Manage account' or find your API key")
    print("4. Copy your API key (44 characters)\n")
    
    token = input("Paste your IBM Quantum API key here: ").strip()
    
    if not token:
        print("Error: No token provided")
        return False
    
    try:
        # Save credentials locally (new platform uses ibm_quantum_platform)
        QiskitRuntimeService.save_account(
            token=token,
            overwrite=True,
            set_as_default=True
        )
        print("\n✓ Credentials saved successfully!")
        print("You can now run experiments on IBM Quantum hardware.\n")
        
        # Test connection
        service = QiskitRuntimeService()
        backends = service.backends()
        print(f"Available backends ({len(backends)}):")
        for b in backends[:5]:
            print(f"  - {b.name}: {b.num_qubits} qubits")
        if len(backends) > 5:
            print(f"  ... and {len(backends)-5} more")
        
        return True
    except Exception as e:
        print(f"\nError saving credentials: {e}")
        return False


def list_backends():
    """List available IBM Quantum backends"""
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        
        print("\n" + "="*60)
        print("Available IBM Quantum Backends")
        print("="*60 + "\n")
        
        # Separate simulators and real devices
        simulators = [b for b in backends if getattr(b, 'simulator', False)]
        real_devices = [b for b in backends if not getattr(b, 'simulator', False)]
        
        print("Real Quantum Computers:")
        for b in sorted(real_devices, key=lambda x: x.num_qubits):
            try:
                status = "online" if b.status().operational else "offline"
                pending = b.status().pending_jobs
                print(f"  {b.name:25} | {b.num_qubits:3} qubits | {status:7} | {pending} jobs queued")
            except:
                print(f"  {b.name:25} | {b.num_qubits:3} qubits")
        
        if simulators:
            print("\nSimulators:")
            for b in simulators:
                print(f"  {b.name:25} | {b.num_qubits:3} qubits")
        
        print("\nRecommended for your 4-qubit VQC:")
        print("  - ibm_brisbane (127 qubits) - Available on free tier")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nHave you set up your credentials? Run:")
        print("  python run_on_ibm_quantum.py --setup")


class IBMQuantumVQC:
    """VQC Classifier that runs on IBM Quantum hardware"""
    
    def __init__(self, n_qubits=4, n_features=20, backend_name=None):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_qubits)
        self.backend_name = backend_name
        
        # Connect to IBM Quantum Platform
        self.service = QiskitRuntimeService()
        
        if backend_name:
            self.backend = self.service.backend(backend_name)
        else:
            # Use least busy backend with >= 4 qubits
            self.backend = self.service.least_busy(
                min_num_qubits=n_qubits,
                operational=True
            )
        
        print(f"Using backend: {self.backend.name}")
        print(f"  Qubits: {self.backend.num_qubits}")
        try:
            print(f"  Pending jobs: {self.backend.status().pending_jobs}")
        except:
            pass
    
    def extract_features(self, img_path):
        """Extract 20 features from blood cell image"""
        try:
            img = imread_collection([img_path])[0]
            
            if len(img.shape) == 3:
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img
            
            img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            features = []
            
            # Intensity statistics
            features.extend([
                np.mean(img_normalized),
                np.std(img_normalized),
                np.median(img_normalized),
                np.percentile(img_normalized, 25),
                np.percentile(img_normalized, 75)
            ])
            
            # GLCM texture
            img_uint8 = (img_normalized * 255).astype(np.uint8)
            glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0]
            ])
            
            # Morphology
            thresh = img_normalized > np.mean(img_normalized)
            labeled = label(thresh)
            if labeled.max() > 0:
                props = regionprops(labeled)[0]
                features.extend([props.area / (64 * 64), props.eccentricity, props.solidity, props.extent])
            else:
                features.extend([0.5, 0.5, 0.5, 0.5])
            
            # Edge features
            edges = sobel(img_normalized)
            features.extend([np.mean(edges), np.std(edges), np.max(edges)])
            
            # FFT features
            fft = np.fft.fft2(img_normalized)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            features.extend([np.mean(magnitude), np.std(magnitude), np.max(magnitude)])
            
            return np.array(features[:self.n_features])
            
        except Exception as e:
            return np.random.randn(self.n_features) * 0.1
    
    def load_data(self, dataset_folder, max_samples_per_class=50):
        """Load blood cell data - using smaller sample for hardware runs"""
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
        """Preprocess data for quantum circuit"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        self.feature_min = X_train_pca.min(axis=0)
        self.feature_max = X_train_pca.max(axis=0)
        
        X_train_final = np.zeros_like(X_train_pca)
        X_test_final = np.zeros_like(X_test_pca)
        
        for i in range(self.n_qubits):
            range_i = self.feature_max[i] - self.feature_min[i] + 1e-8
            X_train_final[:, i] = (X_train_pca[:, i] - self.feature_min[i]) / range_i * np.pi
            X_test_final[:, i] = (X_test_pca[:, i] - self.feature_min[i]) / range_i * np.pi
            X_test_final[:, i] = np.clip(X_test_final[:, i], 0, np.pi)
        
        print(f"PCA variance explained: {sum(self.pca.explained_variance_ratio_)*100:.1f}%")
        return X_train_final, X_test_final
    
    def compute_kernel_matrix(self, X1, X2=None):
        """Compute quantum kernel matrix on IBM hardware"""
        from qiskit import QuantumCircuit
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        
        if X2 is None:
            X2 = X1
        
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=2, entanglement='full')
        
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))
        
        # Build all circuits
        circuits = []
        indices = []
        
        print(f"  Building {n1 * n2} kernel circuits...")
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                # Create fidelity circuit
                qc = QuantumCircuit(self.n_qubits)
                
                # Encode x1
                bound_fm1 = feature_map.assign_parameters(x1)
                qc.compose(bound_fm1, inplace=True)
                
                # Encode x2 (inverse)
                bound_fm2 = feature_map.assign_parameters(x2)
                qc.compose(bound_fm2.inverse(), inplace=True)
                
                qc.measure_all()
                circuits.append(qc)
                indices.append((i, j))
        
        # Transpile for the backend using pass manager
        print(f"  Transpiling circuits for {self.backend.name}...")
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        transpiled = pm.run(circuits)
        
        # Run on IBM Quantum with V2 Sampler
        print(f"  Submitting job to {self.backend.name}...")
        print("  (This may take a while depending on queue)")
        
        sampler = Sampler(backend=self.backend)
        # V2 Sampler takes list of (circuit, parameter_values, shots) tuples
        pubs = [(qc,) for qc in transpiled]
        job = sampler.run(pubs, shots=1024)
        print(f"  Job ID: {job.job_id()}")
        print("  Waiting for results...")
        result = job.result()
        
        # Extract fidelities from measurement outcomes (V2 API)
        for idx, (i, j) in enumerate(indices):
            pub_result = result[idx]
            # Get counts from bit array
            counts = pub_result.data.meas.get_counts()
            total_shots = sum(counts.values())
            # Fidelity is probability of measuring |0...0>
            zero_state = '0' * self.n_qubits
            kernel_matrix[i, j] = counts.get(zero_state, 0) / total_shots
        
        return kernel_matrix
    
    def train_and_predict(self, X_train, y_train, X_test):
        """Train SVM with quantum kernel and predict"""
        print("\nComputing training kernel matrix on IBM Quantum...")
        start = time.time()
        K_train = self.compute_kernel_matrix(X_train)
        train_time = time.time() - start
        print(f"  Training kernel computed in {train_time:.1f}s")
        
        print("\nTraining SVM classifier...")
        svm = SVC(kernel='precomputed', C=1.0)
        svm.fit(K_train, y_train)
        
        print("\nComputing test kernel matrix on IBM Quantum...")
        start = time.time()
        K_test = self.compute_kernel_matrix(X_test, X_train)
        test_time = time.time() - start
        print(f"  Test kernel computed in {test_time:.1f}s")
        
        predictions = svm.predict(K_test)
        return predictions, train_time + test_time


def run_ibm_experiment(dataset_folder, backend_name=None, n_samples=25):
    """Run VQC experiment on IBM Quantum hardware"""
    
    print("\n" + "="*60)
    print("Running VQC on IBM Quantum Hardware")
    print("="*60)
    
    # Smaller sample size for hardware (costs and queue time)
    classifier = IBMQuantumVQC(n_qubits=4, n_features=20, backend_name=backend_name)
    
    # Load data
    X, y = classifier.load_data(dataset_folder, max_samples_per_class=n_samples)
    
    if len(X) == 0:
        print("No data loaded!")
        return
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess
    X_train, X_test = classifier.preprocess(X_train, X_test)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train and predict on IBM hardware
    predictions, total_time = classifier.train_and_predict(X_train, y_train, X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n" + "="*60)
    print("Results (IBM Quantum Hardware)")
    print("="*60)
    print(f"Backend: {classifier.backend.name}")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Quantum Compute Time: {total_time:.1f}s")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Healthy', 'AML']))
    
    # Save results
    results = {
        'backend': classifier.backend.name,
        'accuracy': float(accuracy),
        'quantum_time': float(total_time),
        'n_samples': n_samples
    }
    
    with open('results_ibm_quantum.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results_ibm_quantum.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VQC on IBM Quantum")
    parser.add_argument("--setup", action="store_true", help="Set up IBM Quantum credentials")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    parser.add_argument("--backend", type=str, default=None, help="Specify backend name (e.g., ibm_brisbane)")
    parser.add_argument("--samples", type=int, default=25, help="Samples per class (default: 25)")
    args = parser.parse_args()
    
    if args.setup:
        setup_ibm_quantum()
    elif args.list_backends:
        list_backends()
    else:
        dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at: {dataset_path}")
            print("Please update the dataset_path variable.")
        else:
            run_ibm_experiment(dataset_path, backend_name=args.backend, n_samples=args.samples)
