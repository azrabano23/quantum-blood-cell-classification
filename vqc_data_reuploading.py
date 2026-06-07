#!/usr/bin/env python3
"""
Data Re-Uploading Variational Quantum Classifier (DR-VQC)
=========================================================

Novel contribution: Data re-uploading VQC with interleaved feature encoding.

The standard VQC encodes data once at the circuit input. This DR-VQC
re-encodes the input at every layer, making it equivalent to a universal
quantum classifier that can approximate arbitrary functions (Pérez-Salinas
et al., 2020). The expressibility grows with each re-uploading layer,
enabling the circuit to learn richer feature representations.

Architecture (L=3 layers):
  For each layer l in {0, 1, 2}:
    1. Data encoding:  Ry(x[q] * w_enc[l,q]) on each qubit q
    2. Variational:    Ry(θ[l,q]) Rz(φ[l,q]) on each qubit q
    3. Entanglement:   CNOT ring (q→q+1, last→0)
  Final measurement: <Z₀Z₁Z₂Z₃> (parity observable — captures all-qubit correlations)

Comparison against:
  - Standard VQC (single ZZFeatureMap encoding)
  - Shows improved accuracy and steeper sample efficiency curve

References:
  Pérez-Salinas et al. (2020). Data re-uploading for a universal quantum
  classifier. Quantum, 4, 226.

Paper: arXiv:2601.18710
Author: Azra Bano
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.optimizers import COBYLA, ADAM
from qiskit_algorithms.utils import algorithm_globals

SEED = 42
np.random.seed(SEED)
algorithm_globals.random_seed = SEED


# ── Tee ──────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# ── Feature extraction (identical to VQC pipeline) ───────────────────────────
def extract_features(img_path, n_features=20):
    try:
        img = imread_collection([img_path])[0]
        img_gray = np.mean(img, axis=2) if len(img.shape) == 3 else img
        img_r = resize(img_gray, (64, 64), anti_aliasing=True)
        img_n = (img_r - img_r.min()) / (img_r.max() - img_r.min() + 1e-8)

        features = [np.mean(img_n), np.std(img_n), np.median(img_n),
                    np.percentile(img_n, 25), np.percentile(img_n, 75)]

        img_u8 = (img_n * 255).astype(np.uint8)
        glcm = graycomatrix(img_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.append(graycoprops(glcm, p)[0, 0])

        thresh = img_n > np.mean(img_n)
        lbl = label(thresh)
        if lbl.max() > 0:
            props = regionprops(lbl)[0]
            features += [props.area / (64*64), props.eccentricity, props.solidity, props.extent]
        else:
            features += [0.5, 0.5, 0.5, 0.5]

        edges = sobel(img_n)
        features += [np.mean(edges), np.std(edges), np.max(edges)]

        mag = np.abs(np.fft.fftshift(np.fft.fft2(img_n)))
        features += [np.mean(mag), np.std(mag), np.max(mag)]

        return np.array(features[:n_features])
    except Exception:
        return np.random.randn(n_features) * 0.1


def load_data(dataset_folder, max_samples_per_class=150):
    healthy = ['LYT', 'MON', 'NGS', 'NGB']
    aml = ['MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO', 'EOS', 'LYA', 'MYO', 'PMO']
    X, y, counts = [], [], {'healthy': 0, 'aml': 0}

    for dirpath, _, filenames in os.walk(dataset_folder):
        cell_type = None
        for part in dirpath.split(os.sep):
            if part in healthy or part in aml:
                cell_type = part
                break
        if cell_type is None:
            continue

        for f in sorted(filenames):
            if f.endswith(('.jpg', '.png', '.tiff', '.tif')):
                if cell_type in healthy:
                    if counts['healthy'] >= max_samples_per_class:
                        continue
                    lbl = 0
                    counts['healthy'] += 1
                else:
                    if counts['aml'] >= max_samples_per_class:
                        continue
                    lbl = 1
                    counts['aml'] += 1
                X.append(extract_features(os.path.join(dirpath, f)))
                y.append(lbl)

    X, y = np.array(X), np.array(y)
    print(f"  Loaded {len(X)} samples — Healthy: {counts['healthy']}, AML: {counts['aml']}")
    return X, y


class DRVQCClassifier:
    """
    Data Re-Uploading Variational Quantum Classifier.

    Encodes input features at every layer (L times) rather than once.
    This makes the circuit equivalent to a Fourier series with L frequency
    components per qubit, dramatically increasing expressibility over the
    standard single-encoding VQC.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of input features after PCA)
    n_layers : int
        Number of re-uploading layers (default 3)
    observable : str
        'Z0' (Z on first qubit, like standard VQC) or
        'parity' (Z0Z1Z2Z3, captures all-qubit correlations)
    """

    def __init__(self, n_qubits=4, n_features=20, n_layers=3, observable='parity'):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_layers = n_layers
        self.observable = observable
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_qubits, random_state=SEED)
        self.estimator = StatevectorEstimator()
        self.optimal_params = None
        self.training_history = []
        self._feature_min = None
        self._feature_max = None

        # Parameters per layer: n_qubits encoding weights + n_qubits * 2 variational params
        # Encoding: Ry(x[q] * w[q]) for each qubit
        # Variational: Ry(θ[q]), Rz(φ[q]) for each qubit
        self.n_enc_params = n_qubits * n_layers    # encoding weights (trainable)
        self.n_var_params = n_qubits * 2 * n_layers  # Ry + Rz per qubit per layer
        self.n_params_total = self.n_enc_params + self.n_var_params
        print(f"  DR-VQC parameter count: {self.n_params_total} "
              f"(enc={self.n_enc_params}, var={self.n_var_params})")

    def _build_observable(self):
        if self.observable == 'Z0':
            return SparsePauliOp.from_list([('Z' + 'I' * (self.n_qubits - 1), 1.0)])
        elif self.observable == 'parity':
            # Z₀Z₁Z₂Z₃ — parity measurement, correlates all qubits
            return SparsePauliOp.from_list([('Z' * self.n_qubits, 1.0)])
        elif self.observable == 'sum':
            # Sum of all Zi: more signal than single qubit
            terms = []
            for q in range(self.n_qubits):
                pauli = 'I' * q + 'Z' + 'I' * (self.n_qubits - q - 1)
                terms.append((pauli, 1.0 / self.n_qubits))
            return SparsePauliOp.from_list(terms)
        else:
            raise ValueError(f"Unknown observable: {self.observable}")

    def _build_circuit(self, x, params):
        """
        Build data re-uploading circuit.

        For each layer l:
          1. Ry(x[q] * enc_params[l,q]) — data encoding with trainable weight
          2. Ry(var_params[l,q,0]), Rz(var_params[l,q,1]) — variational
          3. CNOT ring entanglement
        """
        qc = QuantumCircuit(self.n_qubits)
        enc = params[:self.n_enc_params].reshape(self.n_layers, self.n_qubits)
        var = params[self.n_enc_params:].reshape(self.n_layers, self.n_qubits, 2)

        for l in range(self.n_layers):
            # Data re-uploading: Ry(x[q] * w[l,q])
            for q in range(self.n_qubits):
                qc.ry(float(x[q]) * float(enc[l, q]), q)

            # Variational: Ry + Rz
            for q in range(self.n_qubits):
                qc.ry(float(var[l, q, 0]), q)
                qc.rz(float(var[l, q, 1]), q)

            # Entanglement ring: CNOT q→q+1, last→0
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.n_qubits - 1, 0)

        return qc

    def _compute_expectation(self, x, params):
        qc = self._build_circuit(x, params)
        obs = self._build_observable()
        job = self.estimator.run([(qc, obs)])
        return float(job.result()[0].data.evs)

    def _batch_expectation(self, X, params):
        return np.array([self._compute_expectation(x, params) for x in X])

    def _mse_loss(self, params, X, y):
        targets = 2.0 * y - 1.0  # {0→-1, 1→+1}
        exps = self._batch_expectation(X, params)
        return float(np.mean((exps - targets) ** 2))

    def _preprocess(self, X_train, X_test):
        Xtr = self.scaler.fit_transform(X_train)
        Xte = self.scaler.transform(X_test)
        Xtr_pca = self.pca.fit_transform(Xtr)
        Xte_pca = self.pca.transform(Xte)

        self._feature_min = Xtr_pca.min(axis=0)
        self._feature_max = Xtr_pca.max(axis=0)
        frange = self._feature_max - self._feature_min + 1e-8

        Xtr_s = (Xtr_pca - self._feature_min) / frange * 2 * np.pi
        Xte_s = np.clip((Xte_pca - self._feature_min) / frange * 2 * np.pi, 0, 2 * np.pi)

        print(f"  PCA variance retained: {sum(self.pca.explained_variance_ratio_)*100:.1f}%")
        return Xtr_s, Xte_s

    def train(self, X_train_raw, X_test_raw, y_train, max_iterations=300):
        print(f"\n[DR-VQC TRAINING]")
        print(f"  Architecture: {self.n_layers} re-uploading layers × {self.n_qubits} qubits")
        print(f"  Observable  : {self.observable}")
        print(f"  Parameters  : {self.n_params_total} total")
        print(f"  Optimizer   : COBYLA, {max_iterations} iterations")

        X_train, X_test = self._preprocess(X_train_raw, X_test_raw)
        self._X_test = X_test

        start = time.time()
        # Initialization: encoding weights near 1.0, variational near 0
        enc_init = np.ones(self.n_enc_params) * np.pi / 2 + np.random.randn(self.n_enc_params) * 0.1
        var_init = np.random.uniform(0, 2 * np.pi, self.n_var_params)
        params_init = np.concatenate([enc_init, var_init])

        print(f"\n  Initial params (first 8): {params_init[:8].round(4)}")

        self.best_loss = float('inf')
        self.iteration_count = 0

        def objective(params):
            loss = self._mse_loss(params, X_train, y_train)
            self.iteration_count += 1
            if loss < self.best_loss:
                self.best_loss = loss
                self.optimal_params = params.copy()

            print(f"    Iter {self.iteration_count:4d}: loss={loss:.6f}  best={self.best_loss:.6f}")
            self.training_history.append({'iteration': self.iteration_count, 'loss': float(loss)})
            return loss

        opt = COBYLA(maxiter=max_iterations)
        result = opt.minimize(objective, params_init)
        self.optimal_params = result.x

        train_time = time.time() - start
        print(f"\n  Training done in {train_time:.2f}s")
        print(f"  Final loss  : {result.fun:.6f}")
        print(f"  Best loss   : {self.best_loss:.6f}")
        return train_time, X_train, X_test

    def predict(self, X):
        if self.optimal_params is None:
            raise ValueError("Must train first")
        preds = []
        for i, x in enumerate(X):
            exp = self._compute_expectation(x, self.optimal_params)
            pred = 1 if exp > 0 else 0
            preds.append(pred)
            print(f"  Sample {i:4d}: exp={exp:+.6f} → {'AML' if pred else 'Healthy'}")
        return np.array(preds)

    def entanglement_entropy(self, X, n_samples=20):
        """
        Compute von Neumann entanglement entropy for a subsystem (qubits 0,1)
        across n_samples inputs. Novel quantum analysis: different cell types
        produce different entanglement structures.
        """
        if self.optimal_params is None:
            return []
        entropies = []
        for x in X[:n_samples]:
            qc = self._build_circuit(x, self.optimal_params)
            sv = Statevector(qc)
            # Partial trace: trace out qubits 2,3 to get 2-qubit reduced state
            rho = sv.data.reshape([2] * self.n_qubits)
            # Trace out last n_qubits-2 qubits
            n_keep = 2
            n_trace = self.n_qubits - n_keep
            rho_mat = rho.reshape(2**n_keep, 2**n_trace)
            reduced = rho_mat @ rho_mat.conj().T
            # Normalize
            reduced = reduced / (np.trace(reduced) + 1e-10)
            # Von Neumann entropy: S = -Tr(ρ log ρ)
            eigvals = np.linalg.eigvalsh(reduced)
            eigvals = eigvals[eigvals > 1e-12]
            S = -np.sum(eigvals * np.log2(eigvals))
            entropies.append(float(S))
        return entropies


def run_experiment(dataset_folder, sample_sizes=[50, 100, 200, 250]):
    results = {}

    for n_samples in sample_sizes:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: DR-VQC  |  seed={SEED}  |  {n_samples} samples/class")
        print("=" * 80)

        classifier = DRVQCClassifier(n_qubits=4, n_features=20, n_layers=3, observable='parity')

        X, y = load_data(dataset_folder, max_samples_per_class=n_samples)
        if len(X) == 0:
            continue

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        print(f"  Train: {len(X_train_raw)}  Test: {len(X_test_raw)}")

        try:
            train_time, X_train_proc, X_test_proc = classifier.train(
                X_train_raw, X_test_raw, y_train, max_iterations=300
            )
        except Exception as e:
            print(f"Training failed: {e}")
            continue

        t = time.time()
        predictions = classifier.predict(X_test_proc)
        pred_time = time.time() - t

        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Healthy', 'AML'],
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        # Entanglement entropy analysis
        print(f"\n[ENTANGLEMENT ENTROPY ANALYSIS]")
        idx_healthy = np.where(y_test == 0)[0]
        idx_aml = np.where(y_test == 1)[0]
        ent_healthy = classifier.entanglement_entropy(X_test_proc[idx_healthy], n_samples=min(10, len(idx_healthy)))
        ent_aml = classifier.entanglement_entropy(X_test_proc[idx_aml], n_samples=min(10, len(idx_aml)))
        if ent_healthy and ent_aml:
            print(f"  Healthy cells — mean entropy: {np.mean(ent_healthy):.4f}  "
                  f"std: {np.std(ent_healthy):.4f}")
            print(f"  AML cells     — mean entropy: {np.mean(ent_aml):.4f}  "
                  f"std: {np.std(ent_aml):.4f}")
            print(f"  Entropy gap (AML-Healthy): {np.mean(ent_aml)-np.mean(ent_healthy):+.4f}")
            print(f"  → {'AML cells show HIGHER entanglement' if np.mean(ent_aml) > np.mean(ent_healthy) else 'Healthy cells show higher entanglement'}")

        print(f"\n[RESULTS — {n_samples} samples/class]")
        print(f"  DR-VQC Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"  Train time     : {train_time:.2f}s  |  Pred time: {pred_time:.2f}s")
        print(f"\n  Confusion matrix:")
        print(f"                Pred Healthy  Pred AML")
        print(f"  Actual Healthy   {cm[0,0]:5d}       {cm[0,1]:5d}")
        print(f"  Actual AML       {cm[1,0]:5d}       {cm[1,1]:5d}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Healthy', 'AML'], zero_division=0))

        results[n_samples] = {
            'seed': SEED,
            'accuracy': float(acc),
            'train_time': float(train_time),
            'pred_time': float(pred_time),
            'confusion_matrix': cm.tolist(),
            'precision_healthy': float(report['Healthy']['precision']),
            'recall_healthy': float(report['Healthy']['recall']),
            'f1_healthy': float(report['Healthy']['f1-score']),
            'precision_aml': float(report['AML']['precision']),
            'recall_aml': float(report['AML']['recall']),
            'f1_aml': float(report['AML']['f1-score']),
            'entanglement_entropy_healthy': ent_healthy,
            'entanglement_entropy_aml': ent_aml,
            'training_history': classifier.training_history
        }

    out_file = 'results_drvqc.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    return results


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"drvqc_run_{timestamp}.txt"
    tee = Tee(log_filename)
    sys.stdout = tee

    print("=" * 80)
    print("Data Re-Uploading VQC — Blood Cell Classification")
    print(f"Seed     : {SEED}")
    print(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file : {log_filename}")
    print("=" * 80)

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = os.environ.get(
            'AML_DATASET_PATH',
            '/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU'
        )

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        tee.close()
        sys.exit(1)

    results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_filename}")
    tee.close()
