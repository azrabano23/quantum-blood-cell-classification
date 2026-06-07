#!/usr/bin/env python3
"""
Run All 4 Paper Methods with IBM Quantum Hardware Support
=========================================================

Runs all 4 classifiers from arXiv:2601.18710, with the VQC optionally
executing on real IBM quantum hardware via qiskit-ibm-runtime.

Methods:
  1. CNN                   — classical,         target 98.4%  (250 samples/class)
  2. Dense NN              — classical,         target 92.0%  (250 samples/class)
  3. Equilibrium Propagation — quantum-inspired, target 86.4%  ( 50 samples/class)
  4. VQC                   — quantum,           target 83.0%  ( 50 samples/class)

VQC execution modes (--mode):
  simulator  FakeBelemV2 local noise model — no IBM account needed (default)
  hardware   Full COBYLA optimization on real IBM QPU
  hybrid     Train on simulator, validate final params on real hardware

Quick start:
  # Local simulator (no IBM account):
  python run_on_hardware.py --dataset /path/to/AML-Cytomorphology_LMU

  # Real hardware:
  python run_on_hardware.py --dataset /path/to/AML-Cytomorphology_LMU \\
      --mode hardware --token <your-ibm-api-key> --instance <crn>

  # Hybrid (train fast, validate on hardware):
  python run_on_hardware.py --dataset /path/to/AML-Cytomorphology_LMU \\
      --mode hybrid --token <your-ibm-api-key>

  # VQC only, hardware mode:
  python run_on_hardware.py --dataset /path/to/AML-Cytomorphology_LMU \\
      --mode hardware --token <key> --vqc-only

IBM account setup:
  1. Log in at https://quantum.ibm.com
  2. Copy your API key from the dashboard
  3. Find your CRN on the Instances page
  4. Either pass --token / --instance flags, or save credentials once:
       from qiskit_ibm_runtime import QiskitRuntimeService
       QiskitRuntimeService.save_account(token='<key>', instance='<crn>')

Dataset:
  AML-Cytomorphology_LMU — https://doi.org/10.7937/tcia.2019.36f5o9ld

Paper: arXiv:2601.18710
Author: Azra Bano
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Blood cell classification — IBM Quantum hardware runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", "-d",
        default=os.environ.get(
            "AML_DATASET_PATH",
            "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU",
        ),
        help="Path to AML-Cytomorphology_LMU dataset directory",
    )
    p.add_argument(
        "--mode", "-m",
        choices=["hardware", "simulator", "hybrid"],
        default="simulator",
        help="VQC execution mode (default: simulator)",
    )
    p.add_argument(
        "--token", "-t",
        default=None,
        help="IBM Quantum API token (uses saved credentials if omitted)",
    )
    p.add_argument(
        "--instance", "-i",
        default=None,
        help="IBM Quantum CRN instance string",
    )
    p.add_argument(
        "--backend", "-b",
        default=None,
        help="Specific IBM backend name (default: least_busy QPU with ≥4 qubits)",
    )
    p.add_argument(
        "--resilience",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="EstimatorV2 resilience level for error mitigation (default: 1)",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Shots per circuit evaluation on hardware (default: 1024)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="COBYLA max iterations for VQC — paper specifies 200 (default: 200)",
    )
    p.add_argument(
        "--no-dd",
        action="store_true",
        help="Disable dynamical decoupling (XY4) error suppression",
    )
    p.add_argument(
        "--vqc-only",
        action="store_true",
        help="Skip classical methods, run VQC only",
    )
    p.add_argument(
        "--skip-vqc",
        action="store_true",
        help="Skip VQC, run classical methods only",
    )
    p.add_argument(
        "--output", "-o",
        default="results_hardware.json",
        help="Output JSON file (default: results_hardware.json)",
    )
    return p


# ---------------------------------------------------------------------------
# Data preprocessing for VQC (paper-exact)
# ---------------------------------------------------------------------------

def preprocess_vqc(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_qubits: int = 4,
    seed: int = 42,
):
    """
    Standardize → PCA (n_qubits components) → rescale to [0, 2π].

    Scaler and PCA are fit on combined train+test data to replicate the
    paper's preprocessing (which does not enforce strict train-only fitting).
    """
    from qiskit_algorithms.utils import algorithm_globals

    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    scaler = StandardScaler()
    pca = PCA(n_components=n_qubits)

    X_all = np.vstack([X_train, X_test])
    X_all_scaled = scaler.fit_transform(X_all)
    X_all_pca = pca.fit_transform(X_all_scaled)

    X_tr_pca = pca.transform(scaler.transform(X_train))
    X_te_pca = pca.transform(scaler.transform(X_test))

    feat_min = X_all_pca.min(axis=0)
    feat_max = X_all_pca.max(axis=0)

    def _rescale(X):
        out = np.empty_like(X)
        for i in range(n_qubits):
            r = feat_max[i] - feat_min[i] + 1e-8
            out[:, i] = np.clip((X[:, i] - feat_min[i]) / r * 2 * np.pi, 0.0, 2 * np.pi)
        return out

    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA variance explained: {var_explained:.1f}%")
    return _rescale(X_tr_pca), _rescale(X_te_pca)


# ---------------------------------------------------------------------------
# Loaders — thin wrappers around run_paper_exact.py
# ---------------------------------------------------------------------------

def _load_run_paper_exact():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_paper_exact as rpe
    return rpe


# ---------------------------------------------------------------------------
# Classical method runners
# ---------------------------------------------------------------------------

def run_cnn(dataset_path: str) -> dict:
    rpe = _load_run_paper_exact()
    print("\n[CNN] Loading 250 images/class (proportional, seed=42)...")
    X, y, counts = rpe.load_dataset_proportional(dataset_path, 250, mode="images", seed=42)
    X = X[:, np.newaxis, :, :]
    print(f"  {len(X)} images — {counts}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    t0 = time.time()
    _, acc = rpe.train_cnn(X_tr, y_tr, X_te, y_te, epochs=1000)
    elapsed = time.time() - t0
    print(f"  CNN accuracy: {acc:.1%}  ({elapsed:.0f}s)")
    return {"accuracy": float(acc), "time": elapsed, "samples_per_class": 250}


def run_dense_nn(dataset_path: str) -> dict:
    rpe = _load_run_paper_exact()
    print("\n[Dense NN] Loading 250 features/class (proportional, seed=2)...")
    X, y, counts = rpe.load_dataset_proportional(dataset_path, 250, mode="features", seed=2)
    print(f"  {len(X)} samples — {counts}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    t0 = time.time()
    _, acc = rpe.train_dense_nn(X_tr, y_tr, X_te, y_te, epochs=10000)
    elapsed = time.time() - t0
    print(f"  Dense NN accuracy: {acc:.1%}  ({elapsed:.0f}s)")
    return {"accuracy": float(acc), "time": elapsed, "samples_per_class": 250}


def run_ep(dataset_path: str) -> dict:
    rpe = _load_run_paper_exact()
    print("\n[EP] Loading 50 features/class (proportional, seed=42)...")
    X, y, counts = rpe.load_dataset_proportional(dataset_path, 50, mode="features", seed=42)
    print(f"  {len(X)} samples — {counts}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    t0 = time.time()
    _, acc = rpe.train_ep(X_tr, y_tr, X_te, y_te, epochs=100)
    elapsed = time.time() - t0
    print(f"  EP accuracy: {acc:.1%}  ({elapsed:.0f}s)")
    return {"accuracy": float(acc), "time": elapsed, "samples_per_class": 50}


# ---------------------------------------------------------------------------
# VQC hardware runner
# ---------------------------------------------------------------------------

def run_vqc_hardware(dataset_path: str, args: argparse.Namespace) -> dict:
    """
    Run VQC on IBM hardware, fake backend, or hybrid mode.

    Data loading: biased sequential os.walk (paper's original approach),
    50 samples/class — matches the existing vqc_classifier.py behaviour.
    VQC mode mapping:
      hardware  → IBMQuantumVQC(mode='hardware')  — full COBYLA on QPU
      simulator → IBMQuantumVQC(mode='simulator') — FakeBelemV2 locally
      hybrid    → IBMQuantumVQC(mode='simulator') + validate_on_hardware()
    """
    from ibm_quantum_vqc import IBMQuantumVQC

    rpe = _load_run_paper_exact()

    print(f"\n[VQC / {args.mode.upper()}] Loading 50 features/class (biased sequential)...")
    X, y, counts = rpe.load_dataset(dataset_path, 50, mode="features")
    print(f"  {len(X)} samples — {counts}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_proc, X_te_proc = preprocess_vqc(X_tr, X_te, n_qubits=4, seed=42)

    # In hybrid mode we train on the simulator first
    vqc_mode = "simulator" if args.mode == "hybrid" else args.mode

    vqc = IBMQuantumVQC(
        n_qubits=4,
        mode=vqc_mode,
        resilience_level=args.resilience,
        default_shots=args.shots,
        enable_dd=not args.no_dd,
    )
    vqc.connect(
        token=args.token,
        instance=args.instance,
        backend_name=args.backend,
    )

    t0 = time.time()
    vqc.train(X_tr_proc, y_tr, max_iterations=args.iterations)
    train_time = time.time() - t0

    sim_preds = vqc.predict(X_te_proc)
    sim_acc = float(accuracy_score(y_te, sim_preds))
    print(f"\n  {'Simulator' if args.mode != 'hardware' else 'Hardware'} accuracy: {sim_acc:.3f}")
    print(
        classification_report(
            y_te, sim_preds,
            target_names=["Healthy", "AML"],
            zero_division=0,
        )
    )

    result = {
        "mode": args.mode,
        "samples_per_class": 50,
        "iterations": args.iterations,
        "resilience_level": args.resilience,
        "shots": args.shots,
        "dynamical_decoupling": not args.no_dd,
        "train_time_seconds": round(train_time, 2),
        "accuracy_simulator": sim_acc,
        "training_history": vqc.training_history,
    }

    if args.mode == "hardware":
        # The single "accuracy" metric is the hardware run itself
        result["accuracy"] = sim_acc
        result["backend"] = vqc.backend.name if vqc.backend else None

    elif args.mode == "hybrid":
        # Re-run final params on real hardware
        hw_acc = vqc.validate_on_hardware(
            X_te_proc, y_te,
            token=args.token,
            instance=args.instance,
            backend_name=args.backend,
        )
        result["accuracy_hardware"] = hw_acc
        result["accuracy"] = hw_acc
        result["backend"] = vqc.backend.name if vqc.backend else None

    else:  # simulator
        result["accuracy"] = sim_acc

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAPER_TARGETS = {
    "cnn":      0.984,
    "dense_nn": 0.920,
    "ep":       0.864,
    "vqc":      0.830,
}


def main():
    args = _build_parser().parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found at: {args.dataset}")
        print("Set --dataset or AML_DATASET_PATH environment variable.")
        print("Download: https://doi.org/10.7937/tcia.2019.36f5o9ld")
        sys.exit(1)

    # ------------------------------------------------------------------
    print("=" * 80)
    print("QUANTUM BLOOD CELL CLASSIFICATION — IBM QUANTUM HARDWARE RUNNER")
    print("Paper: arXiv:2601.18710")
    print("=" * 80)
    print(f"  Dataset  : {args.dataset}")
    print(f"  VQC mode : {args.mode}")
    print(f"  Output   : {args.output}")
    if args.mode in ("hardware", "hybrid"):
        token_info = "provided via --token" if args.token else "using saved credentials"
        print(f"  IBM token: {token_info}")
        print(f"  Resilience level : {args.resilience}")
        print(f"  Shots/evaluation : {args.shots}")
        print(f"  Dynamical decoupling : {'XY4' if not args.no_dd else 'disabled'}")
    print(f"  COBYLA iterations: {args.iterations}")
    print("=" * 80)

    results = {}

    # ── Classical methods ──────────────────────────────────────────────
    if not args.vqc_only:
        print("\n>>> Running classical methods")

        print("\n" + "-" * 60)
        print("[1/4] CNN — target 98.4%")
        print("-" * 60)
        results["cnn"] = run_cnn(args.dataset)

        print("\n" + "-" * 60)
        print("[2/4] Dense NN — target 92.0%")
        print("-" * 60)
        results["dense_nn"] = run_dense_nn(args.dataset)

        print("\n" + "-" * 60)
        print("[3/4] Equilibrium Propagation — target 86.4%")
        print("-" * 60)
        results["ep"] = run_ep(args.dataset)

    # ── VQC ───────────────────────────────────────────────────────────
    if not args.skip_vqc:
        print("\n" + "-" * 60)
        print(f"[4/4] VQC [{args.mode}] — target 83.0%")
        print("-" * 60)
        results["vqc"] = run_vqc_hardware(args.dataset, args)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL RESULTS vs PAPER  (arXiv:2601.18710)")
    print("=" * 80)
    print(f"{'Model':<14} {'Samples/cls':<13} {'Target':<10} {'Achieved':<12} Status")
    print("-" * 65)

    for model in ["cnn", "dense_nn", "ep", "vqc"]:
        if model not in results:
            continue
        r = results[model]
        achieved = r.get("accuracy", 0.0)
        target = PAPER_TARGETS[model]
        diff = achieved - target
        status = "PASS" if diff >= -0.05 else "FAIL"
        backend_note = f"  [{r['backend']}]" if r.get("backend") else ""
        print(
            f"{model.upper():<14} {r['samples_per_class']:<13} "
            f"{target:.1%}     {achieved:.1%}       {status} ({diff:+.1%})"
            f"{backend_note}"
        )

    # ── Save ──────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
