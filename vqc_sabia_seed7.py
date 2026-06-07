#!/usr/bin/env python3
"""
VQC Classifier - Run for Sabia Hassan
Seed: 7
Prints intermediary values to terminal AND saves to vqc_output_seed7.txt
"""

import sys
import os
import numpy as np
import time
import json

# ── Tee: print to terminal AND to file simultaneously ──────────────────────
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

SEED = 7
output_file = f"vqc_output_seed{SEED}.txt"
sys.stdout = Tee(output_file)
print(f"[VQC] Output logging to: {output_file}")
print(f"[VQC] Random seed: {SEED}")
print(f"[VQC] Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

np.random.seed(SEED)
algorithm_globals.random_seed = SEED
print(f"[VQC] numpy seed set to {SEED}, qiskit seed set to {SEED}")


def extract_features(img_path, n_features=20):
    try:
        img = imread_collection([img_path])[0]
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img
        img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
        img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
        features = []
        features += [np.mean(img_norm), np.std(img_norm), np.median(img_norm),
                     np.percentile(img_norm, 25), np.percentile(img_norm, 75)]
        img_uint8 = (img_norm * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.append(graycoprops(glcm, prop)[0, 0])
        thresh = img_norm > np.mean(img_norm)
        labeled = label(thresh)
        if labeled.max() > 0:
            props = regionprops(labeled)[0]
            features += [props.area / (64*64), props.eccentricity, props.solidity, props.extent]
        else:
            features += [0.5, 0.5, 0.5, 0.5]
        edges = sobel(img_norm)
        features += [np.mean(edges), np.std(edges), np.max(edges)]
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(img_norm)))
        features += [np.mean(fft_mag), np.std(fft_mag), np.max(fft_mag)]
        return np.array(features[:n_features])
    except Exception as e:
        print(f"  [WARN] Feature extraction failed: {e}")
        return np.random.randn(n_features) * 0.1


def load_data(dataset_folder, max_samples=250):
    print(f"\n[DATA] Loading from: {dataset_folder}")
    print(f"[DATA] Max samples per class: {max_samples}")
    healthy = ['LYT', 'MON', 'NGS', 'NGB']
    aml     = ['MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO', 'EOS', 'LYA', 'MYO', 'PMO']
    X, y, counts = [], [], {'healthy': 0, 'aml': 0}
    for dirpath, _, filenames in os.walk(dataset_folder):
        parts = dirpath.split(os.sep)
        cell_type = next((p for p in parts if p in healthy + aml), None)
        if cell_type is None:
            continue
        for file in sorted(filenames):
            if not file.endswith(('.jpg', '.png', '.tiff', '.tif')):
                continue
            if cell_type in healthy:
                if counts['healthy'] >= max_samples: continue
                lbl = 0; counts['healthy'] += 1
            else:
                if counts['aml'] >= max_samples: continue
                lbl = 1; counts['aml'] += 1
            X.append(extract_features(os.path.join(dirpath, file)))
            y.append(lbl)
    X, y = np.array(X), np.array(y)
    print(f"[DATA] Loaded {len(X)} samples — Healthy: {counts['healthy']}, AML: {counts['aml']}")
    print(f"[DATA] Feature matrix shape: {X.shape}")
    print(f"[DATA] Feature means (first 5): {X[:, :5].mean(axis=0).round(4)}")
    print(f"[DATA] Feature stds  (first 5): {X[:, :5].std(axis=0).round(4)}")
    return X, y


def preprocess(X_train, X_test, n_qubits=4):
    scaler = StandardScaler()
    pca    = PCA(n_components=n_qubits)
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)
    print(f"\n[PREPROCESS] PCA variance explained per component: {pca.explained_variance_ratio_.round(4)}")
    print(f"[PREPROCESS] Total variance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"[PREPROCESS] PCA train output (first 3 rows):\n{X_tr_pca[:3].round(4)}")
    fmin = X_tr_pca.min(axis=0)
    fmax = X_tr_pca.max(axis=0)
    rng  = fmax - fmin + 1e-8
    X_tr_final = (X_tr_pca - fmin) / rng * 2 * np.pi
    X_te_final = np.clip((X_te_pca - fmin) / rng * 2 * np.pi, 0, 2*np.pi)
    print(f"[PREPROCESS] After [0,2π] scaling — train min: {X_tr_final.min(axis=0).round(3)}, max: {X_tr_final.max(axis=0).round(3)}")
    return X_tr_final, X_te_final, scaler, pca


estimator = StatevectorEstimator()

def build_circuit(x, params, n_qubits=4):
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full')
    ansatz      = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement='full')
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map.assign_parameters(x), inplace=True)
    qc.compose(ansatz.assign_parameters(params), inplace=True)
    return qc

def compute_expectation(x, params, n_qubits=4):
    qc = build_circuit(x, params, n_qubits)
    obs = SparsePauliOp.from_list([('Z' + 'I' * (n_qubits - 1), 1.0)])
    return estimator.run([(qc, obs)]).result()[0].data.evs

def batch_expectation(X, params, n_qubits=4):
    return np.array([compute_expectation(x, params, n_qubits) for x in X])


def train_vqc(X_train, y_train, n_qubits=4, max_iter=200):
    n_params = n_qubits * 3
    initial_params = np.random.uniform(0, 2*np.pi, n_params)
    print(f"\n[TRAIN] Starting VQC training")
    print(f"[TRAIN] Qubits: {n_qubits} | Params: {n_params} | Max iters: {max_iter}")
    print(f"[TRAIN] Initial params (first 4): {initial_params[:4].round(4)}")
    print(f"[TRAIN] Training samples: {len(X_train)}")
    targets = 2 * y_train - 1
    iteration_count = [0]
    best_state = {'loss': float('inf'), 'params': initial_params.copy()}

    def objective(params):
        expectations = batch_expectation(X_train, params, n_qubits)
        loss = float(np.mean((expectations - targets) ** 2))
        iteration_count[0] += 1
        i = iteration_count[0]
        if loss < best_state['loss']:
            best_state['loss'] = loss
            best_state['params'] = params.copy()
        if i % 10 == 0 or i == 1:
            print(f"  [ITER {i:03d}] loss={loss:.5f} | "
                  f"exp_mean={expectations.mean():.4f} | "
                  f"exp_std={expectations.std():.4f} | "
                  f"best_loss={best_state['loss']:.5f}")
        return loss

    t0 = time.time()
    result = COBYLA(maxiter=max_iter).minimize(objective, initial_params)
    train_time = time.time() - t0
    print(f"\n[TRAIN] Completed in {train_time:.2f}s | Final loss: {result.fun:.5f}")
    print(f"[TRAIN] Best loss seen: {best_state['loss']:.5f}")
    print(f"[TRAIN] Optimal params (first 4): {result.x[:4].round(4)}")
    return result.x, train_time


def predict_vqc(X_test, params, n_qubits=4):
    print(f"\n[PREDICT] Running predictions on {len(X_test)} samples...")
    preds, exps = [], []
    for i, x in enumerate(X_test):
        exp = compute_expectation(x, params, n_qubits)
        pred = 1 if exp > 0 else 0
        preds.append(pred)
        exps.append(exp)
        if i < 5 or i == len(X_test)-1:
            print(f"  Sample {i}: <Z0>={exp:.4f} → pred={pred}")
    exps = np.array(exps)
    print(f"[PREDICT] Expectation values — mean: {exps.mean():.4f}, std: {exps.std():.4f}, "
          f"min: {exps.min():.4f}, max: {exps.max():.4f}")
    return np.array(preds)


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        'AML_DATASET_PATH', '/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU')

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("Usage: python vqc_sabia_seed7.py <dataset_path>")
        sys.exit(1)

    N_SAMPLES = 250
    N_QUBITS  = 4

    X, y = load_data(dataset_path, max_samples=N_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"\n[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[SPLIT] Train class balance — Healthy: {(y_train==0).sum()}, AML: {(y_train==1).sum()}")
    print(f"[SPLIT] Test  class balance — Healthy: {(y_test==0).sum()}, AML: {(y_test==1).sum()}")

    X_train_p, X_test_p, _, _ = preprocess(X_train, X_test, n_qubits=N_QUBITS)
    optimal_params, train_time = train_vqc(X_train_p, y_train, n_qubits=N_QUBITS, max_iter=200)

    t0 = time.time()
    preds = predict_vqc(X_test_p, optimal_params, n_qubits=N_QUBITS)
    pred_time = time.time() - t0

    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*70}")
    print(f"[RESULTS] Seed: {SEED}")
    print(f"[RESULTS] Test Accuracy: {acc*100:.2f}%")
    print(f"[RESULTS] Training Time: {train_time:.2f}s")
    print(f"[RESULTS] Prediction Time: {pred_time:.3f}s")
    print(f"\n[RESULTS] Classification Report:")
    print(classification_report(y_test, preds, target_names=['Healthy', 'AML'], zero_division=0))
    print(f"{'='*70}")
    print(f"[VQC] Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[VQC] Output saved to: {output_file}")
    sys.stdout.log.close()
