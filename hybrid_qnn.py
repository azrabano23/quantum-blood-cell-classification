#!/usr/bin/env python3
"""
Hybrid QNN for Blood Cell Classification — Quantum Transfer Learning
=====================================================================

Two-stage approach (quantum transfer learning):
  Stage 1: Train classical CNN extractor to ~98% using standard cross-entropy.
  Stage 2: Freeze CNN weights, plug in quantum head:
             CNN features (512-dim, frozen)
           → angle projection (Linear(512→4) → Tanh → ×π)
           → 4-qubit quantum circuit (AngleEmbedding + BasicEntanglerLayers, 8 params)
           → post-quantum classifier (Linear(4→2))
           Trained with parameter-shift gradients.

Why two-stage:
  End-to-end training fails because randomly-initialized BasicEntanglerLayers
  push expectation values near zero for all inputs — the post-quantum linear
  sees identical inputs and gradients vanish (barren plateau). Pre-training the
  CNN gives discriminative features before the quantum layer is introduced,
  avoiding this problem. This is standard in quantum transfer learning literature.

Paper: arXiv:2601.18710 (novel 5th method — not in original paper)
Author: A. Zrabano
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.io import imread_collection
from skimage.transform import resize, rotate
import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

import pennylane as qml

# ─────────────────────────── Quantum circuit ────────────────────────────────

N_QUBITS = 4
dev = qml.device('lightning.qubit', wires=N_QUBITS)


@qml.qnode(dev, interface='torch', diff_method='parameter-shift')
def quantum_circuit(inputs, weights):
    """4-qubit circuit: AngleEmbedding + BasicEntanglerLayers (2 layers, 8 params)."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


WEIGHT_SHAPES = {"weights": (2, N_QUBITS)}  # 2 layers × 4 qubits = 8 params


# ──────────────────────────── Model components ──────────────────────────────

def _make_cnn_extractor():
    """4-conv-block CNN feature extractor (shared with BloodCellCNN architecture)."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
    )


class CNNClassifier(nn.Module):
    """Stage 1: full classical CNN (extractor + linear head). Trained first."""
    def __init__(self):
        super().__init__()
        self.extractor = _make_cnn_extractor()
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        return self.head(self.extractor(x))

    def get_features(self, x):
        with torch.no_grad():
            return self.extractor(x)


class QuantumHead(nn.Module):
    """Stage 2: quantum head plugged onto frozen CNN features."""
    def __init__(self):
        super().__init__()
        self.angle_proj = nn.Sequential(nn.Linear(512, N_QUBITS), nn.Tanh())
        # Near-zero init avoids barren plateau — expectation values start distinguishable
        qlayer = qml.qnn.TorchLayer(quantum_circuit, WEIGHT_SHAPES)
        torch.nn.init.normal_(qlayer.weights, mean=0.0, std=0.1)
        self.quantum_layer = qlayer
        self.post_quantum = nn.Linear(N_QUBITS, 2)

    def forward(self, features):
        angles = self.angle_proj(features) * np.pi  # Tanh ∈ [-1,1] → [-π, π]
        q_out = self.quantum_layer(angles)           # (B, 4) expectation values
        return self.post_quantum(q_out)              # (B, 2) logits


# ────────────────────────── Data loading helpers ─────────────────────────────

HEALTHY_TYPES = ['LYT', 'MON', 'NGS', 'NGB']
AML_TYPES     = ['MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO', 'EOS', 'LYA', 'MYO', 'PMO']


def _load_img(img_path, size=64):
    try:
        img = imread_collection([img_path])[0]
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        img_r = resize(img, (size, size), anti_aliasing=True)
        return (img_r - img_r.min()) / (img_r.max() - img_r.min() + 1e-8)
    except Exception:
        return np.zeros((size, size))


def load_images(dataset_folder, max_per_class, seed=42):
    """Proportional random sampling of images across all subtypes."""
    all_paths = {'healthy': [], 'aml': []}
    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        dirnames.sort()
        parts = dirpath.split(os.sep)
        ct = next((p for p in parts if p in HEALTHY_TYPES or p in AML_TYPES), None)
        if ct is None:
            continue
        cls = 'healthy' if ct in HEALTHY_TYPES else 'aml'
        for f in sorted(filenames):
            if f.endswith(('.jpg', '.png', '.tiff', '.tif')):
                all_paths[cls].append(os.path.join(dirpath, f))

    rng = np.random.RandomState(seed)
    rng.shuffle(all_paths['healthy'])
    rng.shuffle(all_paths['aml'])

    X, y = [], []
    for path in all_paths['healthy'][:max_per_class]:
        X.append(_load_img(path)); y.append(0)
    for path in all_paths['aml'][:max_per_class]:
        X.append(_load_img(path)); y.append(1)

    X = np.array(X)[:, np.newaxis, :, :]  # (N, 1, 64, 64)
    return X, np.array(y)


# ──────────────────────────── Augmented dataset ──────────────────────────────

class AugDataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        img, label = self.X[idx].clone(), self.y[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                img = torch.flip(img, [-1])
            if np.random.rand() > 0.5:
                img = torch.flip(img, [-2])
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                img_np = img.numpy()[0]
                img_rot = rotate(img_np, angle, mode='reflect', preserve_range=True)
                img = torch.FloatTensor(img_rot).unsqueeze(0)
            if np.random.rand() > 0.5:
                img = torch.clamp(img * np.random.uniform(0.8, 1.2), 0, 1)
        return img, label


# ──────────────────────── Stage 1: pretrain CNN ──────────────────────────────

def pretrain_cnn(X_train, y_train, X_test, y_test,
                 epochs=500, batch_size=16, lr=0.001, device=None):
    """Train classical CNN extractor. Returns trained CNNClassifier."""
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(2)
    model = CNNClassifier().to(device)

    train_ds = AugDataset(X_train, y_train, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    X_test_t = torch.FloatTensor(X_test).to(device)

    best_acc = 0.0
    best_state = None
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 100 == 0:
            print(f"  [CNN pretrain] Epoch {epoch+1}/{epochs}: "
                  f"Acc={acc:.3f}  Best={best_acc:.3f}  ({time.time()-start:.0f}s)")

    if best_state:
        model.load_state_dict(best_state)
    model.to('cpu')  # move to CPU before quantum stage
    model.eval()
    print(f"  CNN pretrain done: Best={best_acc:.1%}")
    return model, best_acc


# ──────────────────────── Stage 2: quantum head ──────────────────────────────

def train_quantum_head(cnn_model, X_train, y_train, X_test, y_test,
                       epochs=200, batch_size=16, lr=0.01, label=''):
    """
    Freeze CNN, extract features, train quantum head with parameter-shift gradients.
    Runs entirely on CPU (PennyLane TorchLayer requirement).
    """
    device = torch.device('cpu')
    cnn_model.eval()

    # Extract frozen CNN features
    print(f"  Extracting CNN features...")
    with torch.no_grad():
        F_train = cnn_model.get_features(torch.FloatTensor(X_train)).numpy()
        F_test  = cnn_model.get_features(torch.FloatTensor(X_test)).numpy()

    torch.manual_seed(42)
    q_head = QuantumHead().to(device)

    F_train_t = torch.FloatTensor(F_train)
    F_test_t  = torch.FloatTensor(F_test)
    y_train_t = torch.LongTensor(y_train)

    train_ds = torch.utils.data.TensorDataset(F_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(q_head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None
    start = time.time()

    for epoch in range(epochs):
        q_head.train()
        epoch_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            out = q_head(feats)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        q_head.eval()
        with torch.no_grad():
            preds = q_head(F_test_t).argmax(dim=1).numpy()
            acc = accuracy_score(y_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in q_head.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            print(f"  [Q-head {label}] Epoch {epoch+1}/{epochs}: "
                  f"Loss={epoch_loss/len(train_loader):.4f}  "
                  f"Acc={acc:.3f}  Best={best_acc:.3f}  ({time.time()-start:.0f}s)")

    if best_state:
        q_head.load_state_dict(best_state)

    q_head.eval()
    with torch.no_grad():
        final_preds = q_head(F_test_t).argmax(dim=1).numpy()

    total = time.time() - start
    print(f"\n  Quantum head ({label}): Best={best_acc:.1%}  ({total:.0f}s)")
    return final_preds, best_acc


# ──────────────────── Public API used by run_paper_exact.py ──────────────────

def train_hybrid_qnn(X_train, y_train, X_test, y_test,
                     epochs=200, batch_size=16, lr=0.001, label='250/class'):
    """
    Two-stage Hybrid QNN (quantum transfer learning).

    Stage 1: Pretrain CNN extractor (500 epochs, MPS/GPU if available).
    Stage 2: Freeze CNN, train quantum head on extracted features (CPU, parameter-shift).
    Returns (predictions, best_accuracy).
    """
    print(f"\n{'='*60}")
    print(f"Hybrid QNN — Quantum Transfer Learning ({label})")
    print(f"  Stage 1: Pretrain CNN (500 epochs)")
    print(f"  Stage 2: Freeze CNN → quantum head ({epochs} epochs, parameter-shift)")
    print(f"  Samples: {len(X_train)} train / {len(X_test)} test")
    print(f"{'='*60}")

    # Stage 1 — pretrain CNN on MPS/GPU
    cnn_model, cnn_acc = pretrain_cnn(X_train, y_train, X_test, y_test,
                                      epochs=500, batch_size=batch_size, lr=lr)

    # Stage 2 — quantum head on CPU
    preds, q_acc = train_quantum_head(cnn_model, X_train, y_train, X_test, y_test,
                                      epochs=epochs, batch_size=batch_size,
                                      lr=0.01, label=label)

    print(f"\nHybrid QNN ({label}): CNN={cnn_acc:.1%} → QNN={q_acc:.1%}")
    return preds, q_acc


# ─────────────────────────────── Standalone main ─────────────────────────────

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
        sys.exit(1)

    print("=" * 70)
    print("Hybrid QNN — Quantum Transfer Learning (arXiv:2601.18710 extension)")
    print("Stage 1: pretrain CNN  |  Stage 2: freeze CNN + train quantum head")
    print("=" * 70)

    results = {}

    # ── Test A: 250 images/class (comparable to CNN baseline) ──
    print("\n[A] Loading 250 images/class (seed=42)...")
    X_250, y_250 = load_images(dataset_path, 250, seed=42)
    print(f"  Loaded {len(X_250)} images: "
          f"{(y_250==0).sum()} healthy, {(y_250==1).sum()} AML")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_250, y_250, test_size=0.2, random_state=42, stratify=y_250
    )
    preds_250, acc_250 = train_hybrid_qnn(
        X_tr, y_tr, X_te, y_te, epochs=200, batch_size=16, label='250/class'
    )
    results['hybrid_qnn_250'] = {'accuracy': float(acc_250), 'samples_per_class': 250}

    # ── Test B: 50 images/class (comparable to VQC baseline) ──
    print("\n[B] Loading 50 images/class (seed=42)...")
    X_50, y_50 = load_images(dataset_path, 50, seed=42)
    print(f"  Loaded {len(X_50)} images: "
          f"{(y_50==0).sum()} healthy, {(y_50==1).sum()} AML")
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_50, y_50, test_size=0.2, random_state=42, stratify=y_50
    )
    preds_50, acc_50 = train_hybrid_qnn(
        X_tr_s, y_tr_s, X_te_s, y_te_s, epochs=200, batch_size=8, label='50/class'
    )
    results['hybrid_qnn_50'] = {'accuracy': float(acc_50), 'samples_per_class': 50}

    # ── Summary ──
    print("\n" + "=" * 70)
    print("HYBRID QNN RESULTS")
    print("=" * 70)
    print(f"  250 images/class:  {acc_250:.1%}  (comparable to CNN at 98%)")
    print(f"   50 images/class:  {acc_50:.1%}  (comparable to VQC at 70-83%)")
    print(f"\n  Better size: {'250/class' if acc_250 >= acc_50 else '50/class'}")

    with open('results_hybrid_qnn.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results_hybrid_qnn.json")
