#!/usr/bin/env python3
"""
Improved Experiments with Leakage-Free Data Handling
====================================================

Runs all classifiers with:
- 32 comprehensive features (shared extractor)
- Proper train/test split BEFORE preprocessing
- Scaler fit on train only (no leakage)

Author: A. Zrabano
"""

import numpy as np
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_extractor import load_dataset

np.random.seed(42)

# ============ Dense NN ============
def run_dense_nn(X_train, X_test, y_train, y_test, epochs=150):
    """Run improved Dense NN with 32 features"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    torch.manual_seed(42)
    
    input_dim = X_train.shape[1]
    
    # Deeper network for more features
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 2)
    )
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    train_time = time.time() - start
    
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()
    
    return preds, train_time


# ============ Equilibrium Propagation ============
def run_ep(X_train, X_test, y_train, y_test, epochs=80):
    """Run improved Equilibrium Propagation"""
    from equilibrium_propagation import EquilibriumPropagationNetwork
    
    np.random.seed(42)
    
    input_dim = X_train.shape[1]
    
    # Network sized for 32 features
    model = EquilibriumPropagationNetwork(
        layer_sizes=[input_dim, 128, 64, 32, 2],
        beta=0.3,
        learning_rate=0.08,
        use_momentum=True,
        momentum=0.9,
        l2_reg=0.0001
    )
    
    start = time.time()
    model.train(X_train, y_train, epochs=epochs)
    train_time = time.time() - start
    
    preds = model.predict(X_test)
    return preds, train_time


# ============ MIT Hybrid QNN ============
def run_mit_hybrid(X_train, X_test, y_train, y_test, epochs=60):
    """Run improved MIT Hybrid QNN with more features"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import pennylane as qml
    
    torch.manual_seed(42)
    
    n_qubits = 6  # More qubits
    n_qlayers = 3
    input_dim = X_train.shape[1]
    
    # Quantum device
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev, interface='torch')
    def quantum_circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for layer in range(n_qlayers):
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_qubits)
            )
            self.q_weights = nn.Parameter(torch.randn(n_qlayers, n_qubits, 3) * 0.1)
            self.post = nn.Sequential(
                nn.Linear(n_qubits, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )
        
        def forward(self, x):
            x = self.pre(x)
            batch_out = []
            for i in range(x.shape[0]):
                result = quantum_circuit(x[i], self.q_weights)
                batch_out.append(torch.stack(result))
            x = torch.stack(batch_out).float()
            return self.post(x)
    
    model = HybridModel()
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()
    
    return preds, train_time


# ============ VQC ============
def run_vqc(X_train, X_test, y_train, y_test, max_iter=400):
    """Run improved VQC with better encoding"""
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.primitives import Sampler
    try:
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_machine_learning.algorithms import VQC
    except ImportError:
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit.algorithms import algorithm_globals
        from qiskit_machine_learning.algorithms import VQC
    
    algorithm_globals.random_seed = 42
    
    n_qubits = X_train.shape[1]  # After PCA
    
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=2,
        entanglement='full'
    )
    
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=5,  # Deeper
        entanglement='full'
    )
    
    optimizer = COBYLA(maxiter=max_iter, rhobeg=0.5)
    sampler = Sampler()
    
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )
    
    start = time.time()
    vqc.fit(X_train, y_train)
    train_time = time.time() - start
    
    preds = vqc.predict(X_test)
    return preds, train_time


def run_all_experiments(dataset_path, sample_sizes=[100, 200]):
    """Run all experiments with proper data handling"""
    
    results = {}
    
    for n_samples in sample_sizes:
        print(f"\n{'='*80}")
        print(f"EXPERIMENTS WITH {n_samples} SAMPLES PER CLASS")
        print(f"{'='*80}")
        
        # Load data with 32 features
        X, y = load_dataset(dataset_path, max_samples_per_class=n_samples, n_features=32)
        
        if len(X) == 0:
            print("No data loaded!")
            continue
        
        # SPLIT FIRST
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Fit scaler on TRAIN ONLY
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results[n_samples] = {}
        
        # ---- Dense NN ----
        print("\n--- Dense NN ---")
        try:
            preds, train_time = run_dense_nn(X_train_scaled, X_test_scaled, y_train, y_test)
            acc = accuracy_score(y_test, preds)
            print(f"Accuracy: {acc:.3f} | Train time: {train_time:.1f}s")
            results[n_samples]['dense_nn'] = {'accuracy': acc, 'train_time': train_time}
        except Exception as e:
            print(f"Failed: {e}")
        
        # ---- Equilibrium Propagation ----
        print("\n--- Equilibrium Propagation ---")
        try:
            preds, train_time = run_ep(X_train_scaled, X_test_scaled, y_train, y_test)
            acc = accuracy_score(y_test, preds)
            print(f"Accuracy: {acc:.3f} | Train time: {train_time:.1f}s")
            results[n_samples]['ep'] = {'accuracy': acc, 'train_time': train_time}
        except Exception as e:
            print(f"Failed: {e}")
        
        # ---- MIT Hybrid QNN ----
        print("\n--- MIT Hybrid QNN ---")
        try:
            preds, train_time = run_mit_hybrid(X_train_scaled, X_test_scaled, y_train, y_test)
            acc = accuracy_score(y_test, preds)
            print(f"Accuracy: {acc:.3f} | Train time: {train_time:.1f}s")
            results[n_samples]['mit_hybrid'] = {'accuracy': acc, 'train_time': train_time}
        except Exception as e:
            print(f"Failed: {e}")
        
        # ---- VQC (with PCA to reduce dims) ----
        print("\n--- VQC (4 qubits via PCA) ---")
        try:
            # PCA to 4 dims for VQC
            pca = PCA(n_components=4)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            # Scale to [-pi/2, pi/2] using train stats
            train_min, train_max = X_train_pca.min(axis=0), X_train_pca.max(axis=0)
            X_train_vqc = np.pi * (X_train_pca - train_min) / (train_max - train_min + 1e-8) - np.pi/2
            X_test_vqc = np.pi * (X_test_pca - train_min) / (train_max - train_min + 1e-8) - np.pi/2
            
            preds, train_time = run_vqc(X_train_vqc, X_test_vqc, y_train, y_test)
            acc = accuracy_score(y_test, preds)
            print(f"Accuracy: {acc:.3f} | Train time: {train_time:.1f}s")
            results[n_samples]['vqc'] = {'accuracy': acc, 'train_time': train_time}
        except Exception as e:
            print(f"Failed: {e}")
    
    # Save results
    with open('results_improved.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for n_samples, methods in results.items():
        print(f"\n{n_samples} samples per class:")
        for method, data in methods.items():
            print(f"  {method}: {data['accuracy']*100:.1f}%")
    
    return results


if __name__ == "__main__":
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    results = run_all_experiments(dataset_path, sample_sizes=[100, 200])
