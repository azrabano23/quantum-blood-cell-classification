#!/usr/bin/env python3
"""
MIT Hybrid Quantum-Classical Neural Network for Blood Cell Classification
=========================================================================

Implements a hybrid quantum-classical neural network based on the approach
described in the Qiskit textbook and MIT quantum machine learning research.

Architecture:
- Classical preprocessing: Dense layer to reduce dimensionality
- Quantum layer: Parameterized quantum circuit
- Classical postprocessing: Dense layer for classification

This combines the strengths of classical neural networks (feature processing)
with quantum circuits (complex feature transformations).

References:
- Qiskit Machine Learning: https://qiskit.org/textbook/ch-machine-learning/
- MIT Quantum Methods for ML

Author: A. Zrabano
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
import pennylane as qml
import os
import time
import json

np.random.seed(42)
torch.manual_seed(42)

class QuantumLayer(nn.Module):
    """
    Quantum layer that can be integrated into PyTorch neural networks
    """
    
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Number of parameters
        self.n_params = n_qubits * n_layers * 3  # 3 rotations per qubit per layer
        
        # Initialize quantum weights
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Create quantum circuit as a QNode
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            # Encode classical data
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                # Rotation layer
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0], 
                           weights[layer, i, 1], 
                           weights[layer, i, 2], wires=i)
                
                # Entanglement layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # Quantum weights as PyTorch parameters
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
    
    def forward(self, x):
        """Forward pass through quantum layer"""
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            result = self.quantum_circuit(x[i], self.quantum_weights)
            outputs.append(torch.stack(result))
        
        return torch.stack(outputs)

class HybridQuantumClassicalNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network
    
    Architecture:
    Classical Input -> Dense -> Quantum Layer -> Dense -> Output
    """
    
    def __init__(self, input_dim=8, n_qubits=4, n_qlayers=2, hidden_dim=16):
        super().__init__()
        
        # Classical preprocessing
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits)
        )
        
        # Quantum layer
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_qlayers)
        
        # Classical postprocessing
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        # Preprocess with classical network
        x = self.pre_net(x)
        
        # Apply quantum transformation
        x = self.quantum_layer(x)
        
        # Convert to float32 (quantum layer returns float64)
        x = x.float()
        
        # Postprocess with classical network
        x = self.post_net(x)
        
        return x

class MITHybridQNNClassifier:
    """
    MIT-style Hybrid Quantum-Classical Classifier
    """
    
    def __init__(self, input_dim=8, n_qubits=4, n_qlayers=2):
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.model = HybridQuantumClassicalNN(input_dim, n_qubits, n_qlayers)
        self.scaler = StandardScaler()
        self.training_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
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
            # Statistical features
            features.append(np.mean(img_normalized))
            features.append(np.std(img_normalized))
            features.append(np.median(img_normalized))
            features.append(np.percentile(img_normalized, 25))
            features.append(np.percentile(img_normalized, 75))
            
            # Texture features
            features.append(graycoprops(glcm, 'contrast')[0, 0])
            features.append(graycoprops(glcm, 'homogeneity')[0, 0])
            features.append(graycoprops(glcm, 'energy')[0, 0])
            
            return np.array(features[:self.input_dim])
            
        except Exception as e:
            return np.random.randn(self.input_dim) * 0.1
    
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
    
    def train(self, X_train, y_train, epochs=50, batch_size=16, learning_rate=0.01):
        """Train hybrid quantum-classical network"""
        
        print(f"\nTraining MIT Hybrid Quantum-Classical Network")
        print(f"Architecture: Classical({self.input_dim}) -> Quantum({self.n_qubits} qubits, {self.n_qlayers} layers) -> Classical(2)")
        print(f"Training samples: {len(X_train)}")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        start_time = time.time()
        
        self.training_history = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_tensor).float().mean().item()
            
            self.training_history.append({
                'epoch': epoch,
                'loss': epoch_loss / len(dataloader),
                'accuracy': accuracy
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {epoch_loss/len(dataloader):.4f}, Accuracy = {accuracy:.3f}")
        
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
        print(f"EXPERIMENT: MIT Hybrid QNN with {n_samples} samples per class")
        print("="*80)
        
        classifier = MITHybridQNNClassifier(input_dim=8, n_qubits=4, n_qlayers=2)
        
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
        
        # Train
        train_time = classifier.train(X_train, y_train, epochs=50, batch_size=16)
        
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
    with open('results_mit_hybrid.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("All experiments completed. Results saved to results_mit_hybrid.json")
    print("="*80)
    
    return results

if __name__ == "__main__":
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
    else:
        results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
