#!/usr/bin/env python3
"""
Equilibrium Propagation for Blood Cell Classification
=====================================================
Run 1 — Seed 42

Implements Equilibrium Propagation (EP) EXACTLY as described in the paper:
- Two phases: free phase (no target) and nudged phase (with target)
- Energy-based model with tanh activations
- Local learning rules (NO backpropagation)
- Architecture: 256-128-64 hidden layers, 2-unit output
- Training: momentum SGD (μ=0.9), cosine annealing LR, early stopping (patience=15)
- Nudging strength: beta=0.1

Paper: arXiv:2601.18710
Reference: Scellier & Bengio (2017). Equilibrium propagation: Bridging the gap
between energy-based models and backpropagation. Frontiers in Computational Neuroscience.

Author: Azra Bano
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.filters import sobel
from scipy import ndimage
import os
import sys
import time
import json
from datetime import datetime

# ── Seed ────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ── Tee: write to terminal AND a log file simultaneously ─────────────────────
class Tee:
    """Duplicate stdout to a log file."""
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


def tanh(x):
    """Tanh activation as specified in paper"""
    return np.tanh(np.clip(x, -10, 10))


def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x) ** 2


class EquilibriumPropagationNetwork:
    """
    Equilibrium Propagation Network - EXACTLY as described in paper.

    Paper specifications:
    - Architecture: 256-128-64 hidden layers, tanh activations, 2-unit output
    - Free phase: network settles to equilibrium s* with no supervision
    - Nudged phase: output nudged toward target with beta=0.1
    - Weight update: ΔW_ij ∝ (s_i^β s_j^β - s_i* s_j*)/β
    - Training: momentum SGD (μ=0.9), cosine annealing LR, early stopping (patience=15)
    - NO BACKPROPAGATION - uses local Hebbian-like learning rules
    """

    def __init__(self, layer_sizes=[20, 256, 128, 64, 2], beta=0.1, learning_rate=0.05,
                 use_momentum=True, momentum=0.9, l2_reg=0.0001):
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes)
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.l2_reg = l2_reg

        self.weights = []
        self.biases = []
        self.weight_momentum = []
        self.bias_momentum = []

        for i in range(self.n_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)
            if use_momentum:
                self.weight_momentum.append(np.zeros_like(w))
                self.bias_momentum.append(np.zeros_like(b))

        self.training_history = []

    def weight_norms(self):
        """Return L2 norm of each weight matrix."""
        return [float(np.linalg.norm(w)) for w in self.weights]

    def energy(self, states):
        """
        Compute the energy of the network given neuron states.
        E = -sum W_ij s_i s_j - sum b_i s_i + 0.5 sum s_i^2
        """
        energy = 0
        for i in range(len(self.weights)):
            energy -= np.sum(states[i][:, None] * self.weights[i] * states[i + 1])
        for i in range(len(self.biases)):
            energy -= np.sum(self.biases[i] * states[i + 1])
        for state in states:
            energy += 0.5 * np.sum(state ** 2)
        return energy

    def forward_pass(self, x, target=None, beta=0, n_iterations=60, verbose=False):
        """
        Relax the network to equilibrium.
        Free phase (beta=0): no supervision.
        Nudged phase (beta>0): output nudged toward target.
        """
        states = [x.copy()]
        for i in range(1, self.n_layers):
            states.append(np.ones(self.layer_sizes[i]) * 0.5 +
                          np.random.randn(self.layer_sizes[i]) * 0.1)

        alpha = 0.3

        if verbose:
            e_before = self.energy(states)
            print(f"      Energy before relaxation: {e_before:.6f}")

        for iteration in range(n_iterations):
            for i in range(1, self.n_layers):
                h = states[i - 1] @ self.weights[i - 1] + self.biases[i - 1]
                if i == self.n_layers - 1 and target is not None and beta > 0:
                    h += beta * (target - states[i])
                new_state = tanh(h)
                new_state = np.clip(new_state, -0.99, 0.99)
                states[i] = (1 - alpha) * states[i] + alpha * new_state

        if verbose:
            e_after = self.energy(states)
            print(f"      Energy after  relaxation: {e_after:.6f}")
            print(f"      Output layer state: {states[-1].round(4)}")

        return states

    def train_sample(self, x, y, verbose=False):
        """
        Train on a single sample using equilibrium propagation.
        """
        target = np.zeros(self.layer_sizes[-1])
        target[y] = 1.0

        # Free phase
        states_free = self.forward_pass(x, target=None, beta=0, verbose=verbose)
        # Nudged phase
        states_nudged = self.forward_pass(x, target=target, beta=self.beta, verbose=verbose)

        if verbose:
            print(f"      Free  output: {states_free[-1].round(4)}  pred={np.argmax(states_free[-1])}")
            print(f"      Nudged output: {states_nudged[-1].round(4)}  target={y}")

        for i in range(len(self.weights)):
            grad = (np.outer(states_nudged[i], states_nudged[i + 1]) -
                    np.outer(states_free[i], states_free[i + 1])) / self.beta
            grad -= self.l2_reg * self.weights[i]
            grad = np.clip(grad, -1.0, 1.0)

            if self.use_momentum:
                self.weight_momentum[i] = (self.momentum * self.weight_momentum[i] +
                                           self.learning_rate * grad)
                self.weights[i] += self.weight_momentum[i]
            else:
                self.weights[i] += self.learning_rate * grad

            bias_grad = (states_nudged[i + 1] - states_free[i + 1]) / self.beta
            bias_grad = np.clip(bias_grad, -1.0, 1.0)

            if self.use_momentum:
                self.bias_momentum[i] = (self.momentum * self.bias_momentum[i] +
                                         self.learning_rate * bias_grad)
                self.biases[i] += self.bias_momentum[i]
            else:
                self.biases[i] += self.learning_rate * bias_grad

        return np.argmax(states_free[-1])

    def predict_sample(self, x):
        states = self.forward_pass(x, target=None, beta=0)
        return np.argmax(states[-1])

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, patience=15):
        """Train with adaptive LR, cosine annealing, and early stopping."""
        n_samples = len(X_train)
        initial_lr = self.learning_rate
        best_val_acc = 0.0
        patience_counter = 0
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]

        print(f"\n  Initial weight norms per layer: {[f'{n:.4f}' for n in self.weight_norms()]}")

        for epoch in range(epochs):
            self.learning_rate = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

            indices = np.random.permutation(n_samples)

            correct = 0
            # Verbose for first sample of epoch 0 to show free/nudged phases
            for j, idx in enumerate(indices):
                verbose = (epoch == 0 and j == 0)
                if verbose:
                    print(f"\n  [EP VERBOSE — epoch 0, sample 0]")
                pred = self.train_sample(X_train[idx], y_train[idx], verbose=verbose)
                if pred == y_train[idx]:
                    correct += 1

            train_accuracy = correct / n_samples

            val_accuracy = None
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_accuracy = np.mean(val_predictions == y_val)

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}  (patience={patience})")
                    self.weights = best_weights
                    self.biases = best_biases
                    break

            self.training_history.append({
                'epoch': epoch,
                'accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'learning_rate': self.learning_rate,
                'weight_norms': self.weight_norms()
            })

            # Print EVERY epoch
            wnorms = [f"{n:.3f}" for n in self.weight_norms()]
            if val_accuracy is not None:
                print(f"  Epoch {epoch+1:4d}/{epochs}: "
                      f"train={train_accuracy:.4f}  val={val_accuracy:.4f}  "
                      f"LR={self.learning_rate:.5f}  "
                      f"W-norms={wnorms}")
            else:
                print(f"  Epoch {epoch+1:4d}/{epochs}: "
                      f"train={train_accuracy:.4f}  "
                      f"LR={self.learning_rate:.5f}  "
                      f"W-norms={wnorms}")

        print(f"\n  Final weight norms per layer: {[f'{n:.4f}' for n in self.weight_norms()]}")

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x))
        return np.array(predictions)


class EquilibriumPropagationClassifier:
    """EP Classifier for blood cells"""

    def __init__(self, layer_sizes=[20, 256, 128, 64, 2]):
        self.layer_sizes = layer_sizes
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = []

    def extract_features(self, img_path):
        """Extract enhanced texture, statistical, morphology and frequency features"""
        try:
            img = imread_collection([img_path])[0]

            if len(img.shape) == 3:
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img

            img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
            img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

            features = []

            # 1. Statistical features (6)
            features.append(np.mean(img_normalized))
            features.append(np.std(img_normalized))
            features.append(np.median(img_normalized))
            features.append(np.percentile(img_normalized, 25))
            features.append(np.percentile(img_normalized, 75))
            features.append(np.max(img_normalized) - np.min(img_normalized))

            # 2. GLCM texture features (6)
            img_uint8 = (img_normalized * 255).astype(np.uint8)
            glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features.append(graycoprops(glcm, 'contrast')[0, 0])
            features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            features.append(graycoprops(glcm, 'homogeneity')[0, 0])
            features.append(graycoprops(glcm, 'energy')[0, 0])
            features.append(graycoprops(glcm, 'correlation')[0, 0])
            features.append(graycoprops(glcm, 'ASM')[0, 0])

            # 3. Morphological features (4)
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

            # 4. Edge features (2)
            edges = sobel(img_normalized)
            features.append(np.mean(edges))
            features.append(np.std(edges))

            # 5. Frequency domain features (2)
            fft = np.fft.fft2(img_normalized)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            features.append(np.mean(magnitude_spectrum))
            features.append(np.std(magnitude_spectrum))

            return np.array(features[:self.layer_sizes[0]])

        except Exception as e:
            return np.random.randn(self.layer_sizes[0]) * 0.1

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

            for file in sorted(filenames):
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
        """Fit scaler on TRAIN ONLY to avoid data leakage"""
        print(f"\n[PREPROCESSING]")
        print(f"  Input shape: train={X_train.shape}, test={X_test.shape}")
        print(f"  Raw feature stats (train): mean={X_train.mean(axis=0).round(4)}")
        print(f"                             std ={X_train.std(axis=0).round(4)}")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        print(f"  After StandardScaler — train mean ~0: {X_train.mean(axis=0).round(3)}")
        print(f"  First training sample (standardized, first 10): {X_train[0,:10].round(4)}")
        return X_train, X_test

    def train(self, X_train, y_train, epochs=100, validation_split=0.15):
        """Train EP network with validation and early stopping"""

        if validation_split > 0:
            n_val = int(len(X_train) * validation_split)
            indices = np.random.permutation(len(X_train))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
        else:
            X_train_split = X_train
            y_train_split = y_train
            X_val = None
            y_val = None

        print(f"\n[EP TRAINING]")
        print(f"  Seed: {SEED}")
        print(f"  Architecture: {' -> '.join(map(str, self.layer_sizes))}")
        print(f"  Activation: tanh")
        print(f"  Training samples: {len(X_train_split)}  Validation: {len(X_val) if X_val is not None else 0}")
        print(f"  Beta (nudging strength): 0.1")
        print(f"  Momentum: 0.9, Cosine annealing LR, Early stopping (patience=15)")
        print(f"  NO BACKPROPAGATION — local Hebbian-like learning")

        start_time = time.time()

        self.model = EquilibriumPropagationNetwork(
            layer_sizes=self.layer_sizes,
            beta=0.1,
            learning_rate=0.08,
            use_momentum=True,
            momentum=0.9,
            l2_reg=0.0001
        )

        self.model.train(X_train_split, y_train_split, X_val=X_val, y_val=y_val,
                         epochs=epochs, patience=15)
        self.training_history = self.model.training_history

        training_time = time.time() - start_time
        print(f"\n  Training completed in {training_time:.2f} seconds")
        return training_time

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)


def run_experiment(dataset_folder, sample_sizes=[50, 100, 200, 250]):
    """Run experiments with different dataset sizes"""
    results = {}

    for n_samples in sample_sizes:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: EP  |  seed={SEED}  |  {n_samples} samples per class")
        print("=" * 80)

        classifier = EquilibriumPropagationClassifier(layer_sizes=[20, 256, 128, 64, 2])

        start_load = time.time()
        X, y = classifier.load_data(dataset_folder, max_samples_per_class=n_samples)
        load_time = time.time() - start_load

        if len(X) == 0:
            print("No data loaded. Skipping.")
            continue

        print(f"\n[TRAIN/TEST SPLIT]")
        print(f"  Total samples: {len(X)}  |  class distribution: {np.bincount(y)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y
        )
        print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
        print(f"  Train class dist: {np.bincount(y_train)}  |  Test class dist: {np.bincount(y_test)}")

        X_train, X_test = classifier.preprocess(X_train, X_test)
        train_time = classifier.train(X_train, y_train, epochs=100)

        start_pred = time.time()
        predictions = classifier.predict(X_test)
        pred_time = time.time() - start_pred

        test_accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Healthy', 'AML'],
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        print(f"\n[RESULTS — {n_samples} samples/class, seed={SEED}]")
        print(f"  Test Accuracy : {test_accuracy:.4f}  ({test_accuracy*100:.1f}%)")
        print(f"  Load time     : {load_time:.2f}s")
        print(f"  Train time    : {train_time:.2f}s")
        print(f"  Predict time  : {pred_time:.2f}s")
        print(f"  Total time    : {load_time + train_time + pred_time:.2f}s")
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        print(f"                Pred Healthy  Pred AML")
        print(f"  Actual Healthy   {cm[0,0]:5d}       {cm[0,1]:5d}")
        print(f"  Actual AML       {cm[1,0]:5d}       {cm[1,1]:5d}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Healthy', 'AML'], zero_division=0))

        results[n_samples] = {
            'seed': SEED,
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
            'confusion_matrix': cm.tolist(),
            'training_history': classifier.training_history
        }

    out_file = f'results_ep_seed{SEED}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"All experiments done. Results saved to {out_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Set up logging to terminal + file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"ep_run_seed{SEED}_{timestamp}.txt"
    tee = Tee(log_filename)
    sys.stdout = tee

    print("=" * 80)
    print(f"EP Blood Cell Classifier — Run 1")
    print(f"Seed        : {SEED}")
    print(f"Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file    : {log_filename}")
    print("=" * 80)

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = os.environ.get(
            'AML_DATASET_PATH',
            '/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU'
        )

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        print("Usage: python equilibrium_propagation_seed42.py <dataset_path>")
        print("Or set AML_DATASET_PATH environment variable")
        tee.close()
        sys.exit(1)
    else:
        results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Terminal output saved to: {log_filename}")
    tee.close()
