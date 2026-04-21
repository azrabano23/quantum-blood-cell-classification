#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner for AML Blood Cell Classification
==============================================================

Runs EP and VQC across multiple random seeds to generate reproducible
aggregate statistics: mean accuracy, std, and standard error of the mean.

Logs all terminal output to a text file and saves results to JSON.

Usage:
    python run_multi_seed.py [dataset_path]
    python run_multi_seed.py [dataset_path] --seeds 42 7 123 456 789

Paper: arXiv:2601.18710
"""

import numpy as np
import os
import sys
import time
import json
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    QISKIT_AVAILABLE = True
except ImportError:
    print("WARNING: Qiskit not found. VQC experiments will be skipped.")
    QISKIT_AVAILABLE = False


# ── Tee: mirror stdout to a log file ────────────────────────────────────────
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


# ── Feature extraction (shared by EP and VQC) ───────────────────────────────
def extract_features(img_path, n_features=20):
    """
    Compute 20 scalar features from a single 64x64 grayscale image.

    All features are derived purely from pixel data — no external annotations.

    Categories:
      1. Intensity statistics  (5): mean, std, median, 25th pct, 75th pct
      2. GLCM texture          (5): contrast, dissimilarity, homogeneity, energy, correlation
      3. Morphology            (4): normalized area, eccentricity, solidity, extent
                                    (computed from above-mean-threshold binary mask)
      4. Edge metrics          (3): mean Sobel response, std, max
      5. FFT frequency         (3): mean magnitude, std magnitude, peak magnitude
    """
    try:
        img = imread_collection([img_path])[0]
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img.astype(float)

        img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
        mn, mx = img_resized.min(), img_resized.max()
        img_norm = (img_resized - mn) / (mx - mn + 1e-8)

        features = []

        # 1. Intensity statistics
        features.append(float(np.mean(img_norm)))
        features.append(float(np.std(img_norm)))
        features.append(float(np.median(img_norm)))
        features.append(float(np.percentile(img_norm, 25)))
        features.append(float(np.percentile(img_norm, 75)))

        # 2. GLCM texture
        img_uint8 = (img_norm * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)
        features.append(float(graycoprops(glcm, 'contrast')[0, 0]))
        features.append(float(graycoprops(glcm, 'dissimilarity')[0, 0]))
        features.append(float(graycoprops(glcm, 'homogeneity')[0, 0]))
        features.append(float(graycoprops(glcm, 'energy')[0, 0]))
        features.append(float(graycoprops(glcm, 'correlation')[0, 0]))

        # 3. Morphological features (from above-mean binary mask)
        thresh = img_norm > np.mean(img_norm)
        labeled = label(thresh)
        if labeled.max() > 0:
            props = regionprops(labeled)[0]
            features.append(float(props.area / (64 * 64)))   # normalized area
            features.append(float(props.eccentricity))        # elongation
            features.append(float(props.solidity))            # convexity
            features.append(float(props.extent))              # bbox fill ratio
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])

        # 4. Edge metrics (Sobel)
        edges = sobel(img_norm)
        features.append(float(np.mean(edges)))
        features.append(float(np.std(edges)))
        features.append(float(np.max(edges)))

        # 5. FFT frequency features
        mag = np.abs(np.fft.fftshift(np.fft.fft2(img_norm)))
        features.append(float(np.mean(mag)))
        features.append(float(np.std(mag)))
        features.append(float(np.max(mag)))

        return np.array(features[:n_features])

    except Exception as e:
        print(f"  [WARNING] Feature extraction failed for {img_path}: {e}")
        return np.zeros(n_features)


def load_data(dataset_folder, max_samples_per_class=250):
    """
    Walk dataset directory and load balanced (healthy vs AML) features.
    """
    healthy_types = {'LYT', 'MON', 'NGS', 'NGB'}
    aml_types     = {'MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO',
                     'EOS', 'LYA', 'MYO', 'PMO'}

    X, y = [], []
    counts = {'healthy': 0, 'aml': 0}

    for dirpath, _, filenames in os.walk(dataset_folder):
        parts = set(dirpath.split(os.sep))
        cell_type = (parts & healthy_types) or (parts & aml_types)
        if not cell_type:
            continue
        cell_type = next(iter(cell_type))

        for fname in sorted(filenames):
            if not fname.lower().endswith(('.jpg', '.png', '.tiff', '.tif')):
                continue
            if cell_type in healthy_types:
                if counts['healthy'] >= max_samples_per_class:
                    continue
                lbl = 0
                counts['healthy'] += 1
            else:
                if counts['aml'] >= max_samples_per_class:
                    continue
                lbl = 1
                counts['aml'] += 1

            feats = extract_features(os.path.join(dirpath, fname))
            X.append(feats)
            y.append(lbl)

    print(f"  Loaded {len(X)} samples — Healthy: {counts['healthy']}, AML: {counts['aml']}")
    return np.array(X), np.array(y)


# ── Equilibrium Propagation ──────────────────────────────────────────────────
def tanh_clip(x):
    return np.tanh(np.clip(x, -10, 10))


class EPNetwork:
    def __init__(self, layer_sizes, beta=0.1, lr=0.08, momentum=0.9, l2=1e-4):
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.weights, self.biases = [], []
        self.w_mom, self.b_mom = [], []
        for i in range(len(layer_sizes) - 1):
            fan = layer_sizes[i] + layer_sizes[i + 1]
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / fan)
            self.weights.append(w)
            self.biases.append(np.zeros(layer_sizes[i + 1]))
            self.w_mom.append(np.zeros_like(w))
            self.b_mom.append(np.zeros(layer_sizes[i + 1]))

    def relax(self, x, target=None, n_iter=60):
        states = [x.copy()]
        for i in range(1, len(self.layer_sizes)):
            states.append(np.ones(self.layer_sizes[i]) * 0.5
                          + np.random.randn(self.layer_sizes[i]) * 0.1)
        alpha = 0.3
        for _ in range(n_iter):
            for i in range(1, len(self.layer_sizes)):
                h = states[i - 1] @ self.weights[i - 1] + self.biases[i - 1]
                if i == len(self.layer_sizes) - 1 and target is not None:
                    h += self.beta * (target - states[i])
                new_s = tanh_clip(h)
                states[i] = (1 - alpha) * states[i] + alpha * np.clip(new_s, -0.99, 0.99)
        return states

    def train_one(self, x, y_int):
        tgt = np.zeros(self.layer_sizes[-1])
        tgt[y_int] = 1.0
        s_free   = self.relax(x, target=None)
        s_nudged = self.relax(x, target=tgt, n_iter=60)
        for i in range(len(self.weights)):
            g = (np.outer(s_nudged[i], s_nudged[i + 1])
                 - np.outer(s_free[i], s_free[i + 1])) / self.beta
            g -= self.l2 * self.weights[i]
            g = np.clip(g, -1.0, 1.0)
            self.w_mom[i] = self.momentum * self.w_mom[i] + self.lr * g
            self.weights[i] += self.w_mom[i]
            bg = np.clip((s_nudged[i + 1] - s_free[i + 1]) / self.beta, -1.0, 1.0)
            self.b_mom[i] = self.momentum * self.b_mom[i] + self.lr * bg
            self.biases[i] += self.b_mom[i]
        return np.argmax(s_free[-1])

    def predict_one(self, x):
        return np.argmax(self.relax(x)[-1])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


def run_ep_single(X_train, y_train, X_test, y_test, epochs=100, patience=15):
    """Train EP and return (accuracy, train_time_s, infer_time_s)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Carve out 15% validation set for early stopping
    n_val = max(1, int(0.15 * len(X_tr)))
    idx = np.random.permutation(len(X_tr))
    X_val, y_val = X_tr[idx[:n_val]], y_train[idx[:n_val]]
    X_tr2, y_tr2 = X_tr[idx[n_val:]], y_train[idx[n_val:]]

    net = EPNetwork([20, 256, 128, 64, 2])
    best_val, best_w, best_b = 0.0, None, None
    wait = 0
    initial_lr = net.lr

    t0 = time.time()
    for ep in range(epochs):
        net.lr = initial_lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        perm = np.random.permutation(len(X_tr2))
        correct = sum(net.train_one(X_tr2[i], y_tr2[i]) == y_tr2[i] for i in perm)
        tr_acc = correct / len(X_tr2)

        val_acc = np.mean(net.predict(X_val) == y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_w = [w.copy() for w in net.weights]
            best_b = [b.copy() for b in net.biases]
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 10 == 0:
            print(f"    EP epoch {ep+1:3d}/{epochs}: train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}  lr={net.lr:.5f}")

        if wait >= patience:
            print(f"    EP early stop at epoch {ep+1}  (best val={best_val:.3f})")
            net.weights, net.biases = best_w, best_b
            break

    train_time = time.time() - t0

    t1 = time.time()
    preds = net.predict(X_te)
    infer_time = time.time() - t1

    acc = float(accuracy_score(y_test, preds))
    print(f"    EP test accuracy: {acc:.4f}  |  train {train_time:.1f}s  infer {infer_time:.3f}s")
    print(classification_report(y_test, preds, target_names=['Healthy', 'AML'], zero_division=0))
    cm = confusion_matrix(y_test, preds)
    print(f"    Confusion matrix:\n{cm}")

    return acc, train_time, infer_time


# ── VQC ──────────────────────────────────────────────────────────────────────
def run_vqc_single(X_train, y_train, X_test, y_test, seed, max_iter=200):
    """Train VQC and return (accuracy, train_time_s, infer_time_s)."""
    if not QISKIT_AVAILABLE:
        print("    Skipping VQC — Qiskit not available.")
        return None, None, None

    algorithm_globals.random_seed = seed

    n_qubits = 4
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    pca = PCA(n_components=n_qubits)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    print(f"    PCA variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    fmin = X_tr_p.min(axis=0)
    fmax = X_tr_p.max(axis=0)
    def scale_to_2pi(Z, mn, mx):
        return (Z - mn) / (mx - mn + 1e-8) * 2 * np.pi
    X_tr_f = scale_to_2pi(X_tr_p, fmin, fmax)
    X_te_f = np.clip(scale_to_2pi(X_te_p, fmin, fmax), 0, 2 * np.pi)

    estimator = StatevectorEstimator()
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full')
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement='full')
    observable = SparsePauliOp.from_list([('Z' + 'I' * (n_qubits - 1), 1.0)])

    def circuit_expectation(x, params):
        qc = QuantumCircuit(n_qubits)
        qc.compose(feature_map.assign_parameters(x), inplace=True)
        qc.compose(ansatz.assign_parameters(params), inplace=True)
        job = estimator.run([(qc, observable)])
        return float(job.result()[0].data.evs)

    n_params = n_qubits * 3  # RealAmplitudes reps=2 on 4 qubits
    init_params = np.random.uniform(0, 2 * np.pi, n_params)
    targets = 2 * y_train - 1  # {0,1} -> {-1,+1}

    best_loss = float('inf')
    best_params = init_params.copy()
    itr_log = []

    def objective(params):
        nonlocal best_loss, best_params
        exps = np.array([circuit_expectation(x, params) for x in X_tr_f])
        loss = float(np.mean((exps - targets) ** 2))
        itr_log.append(loss)
        n = len(itr_log)
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
        if n % 20 == 0:
            print(f"    VQC iter {n:3d}: loss={loss:.4f}  best_loss={best_loss:.4f}")
        return loss

    t0 = time.time()
    optimizer = COBYLA(maxiter=max_iter)
    result = optimizer.minimize(objective, init_params)
    train_time = time.time() - t0

    print(f"    VQC COBYLA final loss: {result.fun:.4f}  (best seen: {best_loss:.4f})")

    t1 = time.time()
    preds = np.array([1 if circuit_expectation(x, best_params) > 0 else 0
                      for x in X_te_f])
    infer_time = time.time() - t1

    acc = float(accuracy_score(y_test, preds))
    print(f"    VQC test accuracy: {acc:.4f}  |  train {train_time:.1f}s  infer {infer_time:.3f}s")
    print(classification_report(y_test, preds, target_names=['Healthy', 'AML'], zero_division=0))
    cm = confusion_matrix(y_test, preds)
    print(f"    Confusion matrix:\n{cm}")

    return acc, train_time, infer_time


# ── Main multi-seed loop ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    # Try several common paths; override with positional arg or env var
    _default_paths = [
        os.environ.get('AML_DATASET_PATH', ''),
        '/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU',
        '/Users/azrabano/Downloads/AML-Cytomorphology_LMU',
        '/Users/azrabano/Downloads/AML-Cytomorphology',
    ]
    _default = next((p for p in _default_paths if p and os.path.isdir(p)), '')
    parser.add_argument('dataset_path', nargs='?', default=_default)
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 7, 123, 456, 789, 13, 99, 314, 2024, 2025])
    parser.add_argument('--samples', type=int, default=250,
                        help='Samples per class')
    parser.add_argument('--skip-vqc', action='store_true',
                        help='Skip VQC (much faster for EP-only runs)')
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found: {args.dataset_path}")
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'multi_seed_run_{timestamp}.txt'
    results_path = f'multi_seed_results_{timestamp}.json'

    tee = Tee(log_path)
    sys.stdout = tee

    print("=" * 70)
    print(f"Multi-Seed Experiment — AML Blood Cell Classification")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Seeds:   {args.seeds}")
    print(f"Samples per class: {args.samples}")
    print(f"Log file: {log_path}")
    print("=" * 70)

    # Load raw features once (same images for all seeds; only splits change)
    print("\n[Loading dataset...]")
    X_all, y_all = load_data(args.dataset_path, max_samples_per_class=args.samples)

    if len(X_all) == 0:
        print("No data loaded. Check dataset path.")
        sys.exit(1)

    ep_results  = []  # list of (acc, train_t, infer_t)
    vqc_results = []

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        np.random.seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
        )
        print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
        print(f"  Train class dist — Healthy: {(y_train==0).sum()}, AML: {(y_train==1).sum()}")
        print(f"  Test  class dist — Healthy: {(y_test==0).sum()},  AML: {(y_test==1).sum()}")

        # EP
        print(f"\n  -- Equilibrium Propagation (seed={seed}) --")
        np.random.seed(seed)
        ep_acc, ep_tr, ep_inf = run_ep_single(X_train, y_train, X_test, y_test)
        ep_results.append({'seed': seed, 'accuracy': ep_acc,
                           'train_time': ep_tr, 'infer_time': ep_inf})

        # VQC
        if not args.skip_vqc:
            print(f"\n  -- VQC (seed={seed}) --")
            np.random.seed(seed)
            vqc_acc, vqc_tr, vqc_inf = run_vqc_single(
                X_train, y_train, X_test, y_test, seed=seed)
            if vqc_acc is not None:
                vqc_results.append({'seed': seed, 'accuracy': vqc_acc,
                                    'train_time': vqc_tr, 'infer_time': vqc_inf})

    # ── Aggregate statistics ──────────────────────────────────────────────────
    def stats(values, label):
        arr = np.array(values)
        n   = len(arr)
        mu  = float(np.mean(arr))
        sd  = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 1 else 0.0
        lo, hi = float(np.min(arr)), float(np.max(arr))
        print(f"\n  {label} ({n} seeds):")
        print(f"    Mean  = {mu:.4f}")
        print(f"    Std   = {sd:.4f}")
        print(f"    SEM   = {sem:.4f}  (95% CI ≈ ±{1.96*sem:.4f})")
        print(f"    Range = [{lo:.4f}, {hi:.4f}]")
        print(f"    Individual: {[round(v,4) for v in arr.tolist()]}")
        return {'mean': mu, 'std': sd, 'sem': sem, 'min': lo, 'max': hi,
                'n': n, 'values': arr.tolist()}

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    ep_accs  = [r['accuracy']   for r in ep_results]
    ep_trs   = [r['train_time'] for r in ep_results]
    ep_infs  = [r['infer_time'] for r in ep_results]

    ep_acc_stats  = stats(ep_accs,  "EP Accuracy")
    ep_tr_stats   = stats(ep_trs,   "EP Train Time (s)")
    ep_inf_stats  = stats(ep_infs,  "EP Infer Time (s)")

    summary = {
        'timestamp': timestamp,
        'seeds': args.seeds,
        'samples_per_class': args.samples,
        'ep': {
            'per_seed': ep_results,
            'accuracy': ep_acc_stats,
            'train_time': ep_tr_stats,
            'infer_time': ep_inf_stats,
        }
    }

    if vqc_results:
        vqc_accs = [r['accuracy']   for r in vqc_results]
        vqc_trs  = [r['train_time'] for r in vqc_results]
        vqc_infs = [r['infer_time'] for r in vqc_results]

        vqc_acc_stats  = stats(vqc_accs,  "VQC Accuracy")
        vqc_tr_stats   = stats(vqc_trs,   "VQC Train Time (s)")
        vqc_inf_stats  = stats(vqc_infs,  "VQC Infer Time (s)")

        summary['vqc'] = {
            'per_seed': vqc_results,
            'accuracy': vqc_acc_stats,
            'train_time': vqc_tr_stats,
            'infer_time': vqc_inf_stats,
        }

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_path}")
    print(f"Log saved to:     {log_path}")

    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    tee.close()


if __name__ == '__main__':
    main()
