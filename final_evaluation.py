#!/usr/bin/env python3
"""
Final Evaluation: Quantum vs Classical Blood Cell Classification
================================================================

EVALUATION METHODOLOGY:
-----------------------
Track A (Image-Level Split): 
  - Random stratified split at image level
  - Preprocessing (scaler/PCA) fit on TRAIN ONLY
  - May overestimate generalization due to unknown patient overlap
  
Track B (Patient-Level Split):
  - NOT POSSIBLE with public AML-Cytomorphology_LMU package
  - Patient IDs were removed via HIPAA-compliant de-identification
  - Would require external metadata from original authors

IMPORTANT CAVEAT (to include in any publication):
"Because the publicly released AML-Cytomorphology_LMU image files do not include 
patient identifiers, we evaluate using an image-level split. This may overestimate 
performance due to potential patient overlap between train and test sets. We 
therefore interpret reported accuracies as optimistic upper bounds; patient-level 
evaluation requires metadata not included in the public release."

Author: A. Zrabano
"""

import numpy as np
import json
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from feature_extractor import load_dataset

np.random.seed(42)

DATASET_PATH = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"

# ============ Optimized Dense NN ============
def run_dense_nn(X_train, X_test, y_train, y_test, epochs=200):
    """Optimized Dense NN"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    torch.manual_seed(42)
    input_dim = X_train.shape[1]
    
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 2)
    )
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()
    
    return preds, time.time() - start


# ============ Optimized Equilibrium Propagation ============
def run_ep(X_train, X_test, y_train, y_test, epochs=200):
    """Best EP configuration found through hyperparameter search"""
    from equilibrium_propagation import EquilibriumPropagationNetwork
    
    np.random.seed(42)
    input_dim = X_train.shape[1]
    
    # Best config: [32, 128, 64, 2] with beta=0.35, lr=0.1
    model = EquilibriumPropagationNetwork(
        layer_sizes=[input_dim, 128, 64, 2],
        beta=0.35,
        learning_rate=0.1,
        use_momentum=True,
        momentum=0.9,
        l2_reg=0.0001
    )
    
    start = time.time()
    model.train(X_train, y_train, epochs=epochs)
    
    return model.predict(X_test), time.time() - start


# ============ Optimized MIT Hybrid QNN ============
def run_mit_hybrid(X_train, X_test, y_train, y_test, epochs=80):
    """Optimized Hybrid QNN with 8 qubits"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import pennylane as qml
    
    torch.manual_seed(42)
    
    n_qubits = 8
    n_qlayers = 4
    input_dim = X_train.shape[1]
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev, interface='torch')
    def quantum_circuit(inputs, weights):
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i] * 0.5, wires=i)
        
        # Variational layers with strong entanglement
        for layer in range(n_qlayers):
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            # Ring entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            # Cross entanglement every other layer
            if layer % 2 == 1:
                for i in range(n_qubits // 2):
                    qml.CZ(wires=[i, i + n_qubits // 2])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_qubits)
            )
            self.q_weights = nn.Parameter(torch.randn(n_qlayers, n_qubits, 3) * 0.3)
            self.post = nn.Sequential(
                nn.Linear(n_qubits, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
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
    optimizer = optim.Adam(model.parameters(), lr=0.008)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()
    
    return preds, time.time() - start


# ============ CNN (Raw Images) ============
def run_cnn(dataset_path, n_samples, epochs=80):
    """CNN on raw images - separate data pipeline"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from skimage.io import imread_collection
    from skimage.transform import resize
    import os
    
    torch.manual_seed(42)
    
    # Load raw images
    print("  Loading images for CNN...")
    healthy_types = ['LYT', 'MON', 'NGS', 'NGB']
    aml_types = ['MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO', 'EOS', 'LYA', 'MYO', 'PMO']
    
    X, y = [], []
    counts = {'healthy': 0, 'aml': 0}
    
    for dirpath, _, filenames in os.walk(dataset_path):
        cell_type = None
        for part in dirpath.split(os.sep):
            if part in healthy_types + aml_types:
                cell_type = part
                break
        if not cell_type:
            continue
        
        for f in filenames:
            if not f.endswith(('.tiff', '.tif')):
                continue
            if cell_type in healthy_types:
                if counts['healthy'] >= n_samples:
                    continue
                label = 0
                counts['healthy'] += 1
            else:
                if counts['aml'] >= n_samples:
                    continue
                label = 1
                counts['aml'] += 1
            
            try:
                img = imread_collection([os.path.join(dirpath, f)])[0]
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2)
                img = resize(img, (64, 64), anti_aliasing=True)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                X.append(img)
                y.append(label)
            except:
                pass
    
    X = np.array(X)[:, np.newaxis, :, :]  # Add channel dim
    y = np.array(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # CNN Model
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 2)
            )
        def forward(self, x):
            return self.fc(self.conv(x))
    
    model = CNN()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
                       batch_size=16, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()
    
    return preds, y_test, time.time() - start


# ============ VQC (Pure Quantum) ============
def run_vqc(X_train, X_test, y_train, y_test, max_iter=500):
    """Variational Quantum Classifier with optimized config"""
    from sklearn.decomposition import PCA
    from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
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
    
    # PCA to 6 qubits
    pca = PCA(n_components=6)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Scale to [-pi/2, pi/2]
    train_min, train_max = X_train_pca.min(axis=0), X_train_pca.max(axis=0)
    X_train_q = np.pi * (X_train_pca - train_min) / (train_max - train_min + 1e-8) - np.pi/2
    X_test_q = np.pi * (X_test_pca - train_min) / (train_max - train_min + 1e-8) - np.pi/2
    
    n_qubits = 6
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    ansatz = EfficientSU2(num_qubits=n_qubits, reps=4, entanglement='circular')
    
    optimizer = COBYLA(maxiter=max_iter, rhobeg=0.5)
    sampler = Sampler()
    
    vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    
    start = time.time()
    vqc.fit(X_train_q, y_train)
    preds = vqc.predict(X_test_q)
    
    return preds, time.time() - start


# ============ SVM Baseline ============
def run_svm(X_train, X_test, y_train, y_test):
    """Classical SVM baseline"""
    start = time.time()
    model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test), time.time() - start


def run_track_a_evaluation(n_samples=200, n_features=32):
    """
    Track A: Image-level split evaluation
    
    CAVEAT: May overestimate performance due to potential patient overlap.
    """
    print("=" * 80)
    print("TRACK A: IMAGE-LEVEL SPLIT EVALUATION")
    print("=" * 80)
    print("\nCAVEAT: Image-level split may overestimate performance due to")
    print("potential patient overlap between train and test sets.\n")
    
    # Load data
    X, y = load_dataset(DATASET_PATH, max_samples_per_class=n_samples, n_features=n_features)
    
    if len(X) == 0:
        print("ERROR: No data loaded!")
        return None
    
    # SPLIT FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {n_features}")
    
    # Fit scaler on TRAIN ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # ---- SVM Baseline ----
    print("\n--- SVM (Classical Baseline) ---")
    preds, train_time = run_svm(X_train_scaled, X_test_scaled, y_train, y_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.1%} | Time: {train_time:.1f}s")
    results['svm'] = {'accuracy': acc, 'time': train_time}
    
    # ---- Dense NN ----
    print("\n--- Dense NN (Classical) ---")
    preds, train_time = run_dense_nn(X_train_scaled, X_test_scaled, y_train, y_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.1%} | Time: {train_time:.1f}s")
    results['dense_nn'] = {'accuracy': acc, 'time': train_time}
    
    # ---- Equilibrium Propagation ----
    print("\n--- Equilibrium Propagation (Bio-inspired) ---")
    preds, train_time = run_ep(X_train_scaled, X_test_scaled, y_train, y_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.1%} | Time: {train_time:.1f}s")
    results['ep'] = {'accuracy': acc, 'time': train_time}
    
    # ---- MIT Hybrid QNN ----
    print("\n--- MIT Hybrid QNN (Quantum) ---")
    preds, train_time = run_mit_hybrid(X_train_scaled, X_test_scaled, y_train, y_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.1%} | Time: {train_time:.1f}s")
    results['mit_hybrid_qnn'] = {'accuracy': acc, 'time': train_time}
    
    # ---- CNN (Raw Images) ----
    print("\n--- CNN (Classical, Raw Images) ---")
    preds_cnn, y_test_cnn, train_time = run_cnn(DATASET_PATH, n_samples)
    acc = accuracy_score(y_test_cnn, preds_cnn)
    print(f"Accuracy: {acc:.1%} | Time: {train_time:.1f}s")
    results['cnn'] = {'accuracy': acc, 'time': train_time}
    
    # ---- VQC (Pure Quantum) ----
    print("\n--- VQC (Pure Quantum, 6 qubits) ---")
    preds, train_time = run_vqc(X_train_scaled, X_test_scaled, y_train, y_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.1%} | Time: {train_time:.1f}s")
    results['vqc'] = {'accuracy': acc, 'time': train_time}
    
    # Summary
    print("\n" + "=" * 80)
    print("TRACK A SUMMARY (Image-Level Split)")
    print("=" * 80)
    for method, data in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        marker = "★" if "qnn" in method or "ep" in method else " "
        print(f"{marker} {method:20s}: {data['accuracy']:6.1%}")
    
    return results


def document_track_b():
    """
    Track B: Patient-level split - NOT POSSIBLE
    """
    print("\n" + "=" * 80)
    print("TRACK B: PATIENT-LEVEL SPLIT")
    print("=" * 80)
    print("""
STATUS: NOT POSSIBLE with public AML-Cytomorphology_LMU package

REASON: Patient identifiers were removed via HIPAA-compliant de-identification 
        process before public release on TCIA.

EVIDENCE:
  - Searched for: *.csv, *.xlsx, *.json, *.xml, metadata files
  - Found only: AML-Cytomorphology.sums (MD5 checksums, no patient mapping)
  - Image filenames are sequential (LYT_0001.tiff, etc.) with no patient grouping
  - TCIA documentation confirms de-identification

ALTERNATIVE: The newer AML-Cytomorphology_MLL_Helmholtz dataset includes patient 
             metadata in a CSV file, but this is a different dataset.

FOR PUBLICATION, USE THIS CAVEAT:
"Because the publicly released AML-Cytomorphology_LMU image files do not include 
patient identifiers, we evaluate using an image-level split. This may overestimate 
performance due to potential patient overlap between train and test sets. We 
therefore interpret reported accuracies as optimistic upper bounds."
""")


if __name__ == "__main__":
    # Run Track A evaluation
    results = run_track_a_evaluation(n_samples=200, n_features=32)
    
    # Document Track B impossibility
    document_track_b()
    
    # Save results
    if results:
        with open('results_final_track_a.json', 'w') as f:
            json.dump({
                'track': 'A (image-level)',
                'caveat': 'May overestimate due to patient overlap',
                'results': {k: {'accuracy': v['accuracy'], 'time': v['time']} 
                           for k, v in results.items()}
            }, f, indent=2)
        print("\nResults saved to results_final_track_a.json")
