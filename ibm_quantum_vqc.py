#!/usr/bin/env python3
"""
IBM Quantum Hardware VQC for Blood Cell Classification
=======================================================

Hardware-ready VQC that runs on real IBM QPUs using qiskit-ibm-runtime.
Uses a single parameterized circuit that is transpiled ONCE to the backend ISA.

Key differences from vqc_classifier.py (local StatevectorEstimator):
  - Uses EstimatorV2 from qiskit_ibm_runtime for hardware execution
  - Builds one fully-parameterized circuit, transpiles to backend ISA once
  - Batches ALL training samples into a single PUB per COBYLA iteration
  - Supports resilience levels (0-2) and dynamical decoupling (XY4)
  - Three execution modes: 'hardware', 'simulator', 'hybrid'

Modes:
  hardware   Full COBYLA optimization on real IBM QPU (authentic, expensive)
  simulator  FakeBelemV2 local noise model — fast testing, no account needed
  hybrid     Train on simulator, validate final circuit on real hardware

Usage:
    from ibm_quantum_vqc import IBMQuantumVQC

    vqc = IBMQuantumVQC(n_qubits=4, mode='hardware', resilience_level=1)
    vqc.connect(token='<your-token>', instance='<crn>')
    vqc.train(X_train, y_train, max_iterations=200)
    preds = vqc.predict(X_test)

    # Hybrid — train on sim, validate on hardware:
    vqc = IBMQuantumVQC(n_qubits=4, mode='simulator')
    vqc.connect()
    vqc.train(X_train, y_train)
    hw_acc = vqc.validate_on_hardware(X_test, y_test, token='...', instance='...')

Paper: arXiv:2601.18710
"""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from sklearn.metrics import accuracy_score


class IBMQuantumVQC:
    """
    VQC that executes on real IBM Quantum hardware via qiskit-ibm-runtime.

    Circuit architecture (paper-exact):
      Feature map : ZZFeatureMap — 4 qubits, reps=2, full entanglement
      Ansatz      : RealAmplitudes — 4 qubits, reps=2, full entanglement
      Parameters  : 12 trainable (4 qubits × 3 layers from reps=2)
      Observable  : <Z ⊗ I ⊗ I ⊗ I>  (Z on qubit 0, identity elsewhere)

    Modes:
      'hardware'   — real IBM QPU via QiskitRuntimeService
      'simulator'  — FakeBelemV2 local noise model (fast, no account needed)
      'hybrid'     — optimize on simulator, validate final params on hardware
    """

    def __init__(
        self,
        n_qubits: int = 4,
        mode: str = "hardware",
        resilience_level: int = 1,
        default_shots: int = 1024,
        enable_dd: bool = True,
    ):
        self.n_qubits = n_qubits
        self.mode = mode
        self.resilience_level = resilience_level
        self.default_shots = default_shots
        self.enable_dd = enable_dd

        self.backend = None
        self.estimator = None
        self.isa_circuit = None
        self.mapped_observable = None
        self.optimal_params = None
        self.training_history = []
        self.best_loss = float("inf")

        # Build once; transpile later when backend is known
        self._base_circuit = self._build_parameterized_circuit()

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_parameterized_circuit(self) -> QuantumCircuit:
        """
        Build fully-parameterized VQC (feature map + ansatz).

        The circuit is composed once and transpiled once to the backend ISA.
        Parameters are bound per COBYLA iteration — no circuit rebuilding.

        Parameter naming:
          'x[0..3]'  — input features (bound per sample)
          'θ[0..11]' — ansatz weights  (optimized by COBYLA)
        """
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement="full",
            parameter_prefix="x",
        )
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=2,
            entanglement="full",
            parameter_prefix="θ",
        )
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        return qc

    # ------------------------------------------------------------------
    # Backend connection & transpilation
    # ------------------------------------------------------------------

    def connect(
        self,
        token: str = None,
        instance: str = None,
        backend_name: str = None,
    ):
        """
        Authenticate with IBM Quantum and transpile circuit to backend ISA.

        Transpilation happens ONCE here; every subsequent iteration reuses the
        already-compiled ISA circuit, which avoids repeated compilation overhead.

        Args:
            token:        IBM Quantum API token (uses saved credentials if None)
            instance:     CRN instance string (optional)
            backend_name: Specific backend name; uses least_busy if None
        """
        if self.mode == "simulator":
            self._setup_fake_backend()
        elif self.mode in ("hardware", "hybrid"):
            self._setup_hardware_backend(token, instance, backend_name)
        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Choose: hardware | simulator | hybrid"
            )

    def _setup_fake_backend(self):
        """FakeBelemV2 — local noise model, no IBM account needed."""
        from qiskit_ibm_runtime import EstimatorV2 as Estimator
        from qiskit_ibm_runtime.fake_provider import FakeBelemV2

        print("[IBMQuantumVQC] Backend: FakeBelemV2 (local noise model)")
        self.backend = FakeBelemV2()
        self._transpile_and_setup(Estimator(mode=self.backend))

    def _setup_hardware_backend(
        self, token: str, instance: str, backend_name: str
    ):
        """Connect to real IBM QPU with resilience + dynamical decoupling."""
        from qiskit_ibm_runtime import (
            EstimatorOptions,
            EstimatorV2 as Estimator,
            QiskitRuntimeService,
        )

        service = (
            QiskitRuntimeService(token=token, instance=instance)
            if token
            else QiskitRuntimeService()
        )

        self.backend = (
            service.backend(backend_name)
            if backend_name
            else service.least_busy(
                simulator=False,
                operational=True,
                min_num_qubits=self.n_qubits,
            )
        )
        print(f"[IBMQuantumVQC] Backend: {self.backend.name}")
        pending = self.backend.status().pending_jobs
        print(f"[IBMQuantumVQC] Queue:   {pending} pending jobs")

        options = EstimatorOptions()
        options.resilience_level = self.resilience_level
        options.default_shots = self.default_shots
        if self.enable_dd:
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = "XY4"

        self._transpile_and_setup(Estimator(mode=self.backend, options=options))

    def _transpile_and_setup(self, estimator):
        """Transpile base circuit to backend ISA and map the observable."""
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = pm.run(self._base_circuit)

        # Z on qubit 0 after layout mapping
        raw_obs = SparsePauliOp("Z" + "I" * (self.n_qubits - 1))
        self.mapped_observable = raw_obs.apply_layout(self.isa_circuit.layout)

        self.estimator = estimator
        n_params = len(self.isa_circuit.parameters)
        depth = self.isa_circuit.depth()
        print(
            f"[IBMQuantumVQC] Circuit transpiled — depth: {depth}, params: {n_params}"
        )

    # ------------------------------------------------------------------
    # Batched expectation values (one job per COBYLA iteration)
    # ------------------------------------------------------------------

    def _sorted_params(self):
        """Return circuit parameters sorted: x-params first, then θ-params."""
        x_ps = sorted(
            [p for p in self.isa_circuit.parameters if p.name.startswith("x")],
            key=lambda p: p.name,
        )
        t_ps = sorted(
            [p for p in self.isa_circuit.parameters if p.name.startswith("θ")],
            key=lambda p: p.name,
        )
        return x_ps, t_ps

    def _build_param_matrix(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Build parameter value matrix of shape (n_samples, n_params).

        Each row encodes [x_i ..., θ ...] for one sample, in the exact
        column order that the ISA circuit's ParameterView expects.
        """
        x_ps, t_ps = self._sorted_params()
        n_samples = len(X)
        n_params = len(self.isa_circuit.parameters)
        all_ps = list(self.isa_circuit.parameters)  # canonical order

        values = np.empty((n_samples, n_params))
        for col, param in enumerate(all_ps):
            if param in x_ps:
                feat_idx = x_ps.index(param) % self.n_qubits
                values[:, col] = X[:, feat_idx]
            else:
                theta_idx = t_ps.index(param)
                values[:, col] = theta[theta_idx]
        return values

    def _batch_expectations(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Evaluate <Z0> for all n_samples in X with fixed ansatz params theta.

        Submits ONE PUB to the estimator — the runtime batches all rows of the
        parameter matrix into a single hardware job, minimising queue overhead.

        Returns: np.ndarray of shape (n_samples,)
        """
        param_values = self._build_param_matrix(X, theta)
        pub = (self.isa_circuit, self.mapped_observable, param_values)
        job = self.estimator.run([pub])
        evs = np.asarray(job.result()[0].data.evs).flatten()
        return evs

    # ------------------------------------------------------------------
    # COBYLA optimization
    # ------------------------------------------------------------------

    def _mse_loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        MSE loss between <Z0> expectation values and ±1 targets (paper-exact).
        Label convention: 0 (healthy) → target -1,  1 (AML) → target +1
        """
        targets = 2 * y - 1
        expectations = self._batch_expectations(X, theta)
        return float(np.mean((expectations - targets) ** 2))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_iterations: int = 200,
    ):
        """
        Optimize ansatz parameters using COBYLA (paper-exact, gradient-free).

        Each COBYLA iteration submits ONE batched job — all training samples
        are packed into a single PUB so the hardware overhead is minimised.

        Args:
            X_train:        Preprocessed features, shape (n_samples, n_qubits)
            y_train:        Binary labels  0 = healthy,  1 = AML
            max_iterations: COBYLA iterations (paper specifies 200)
        """
        if self.estimator is None:
            raise RuntimeError("Call connect() before train().")

        _, t_ps = self._sorted_params()
        n_theta = len(t_ps)
        np.random.seed(42)
        algorithm_globals.random_seed = 42
        initial_theta = np.random.uniform(0, 2 * np.pi, n_theta)

        self.best_loss = float("inf")
        self.optimal_params = initial_theta.copy()
        self.training_history = []
        iteration = [0]

        def objective(theta):
            loss = self._mse_loss(theta, X_train, y_train)
            iteration[0] += 1
            self.training_history.append(
                {"iteration": iteration[0], "loss": float(loss)}
            )
            if loss < self.best_loss:
                self.best_loss = loss
                self.optimal_params = theta.copy()
            if iteration[0] % 10 == 0:
                preds = (self._batch_expectations(X_train, theta) > 0).astype(int)
                train_acc = accuracy_score(y_train, preds)
                print(
                    f"  iter {iteration[0]:>3}/{max_iterations}  "
                    f"loss={loss:.4f}  train_acc={train_acc:.3f}"
                )
            return loss

        backend_label = (
            self.backend.name if self.backend else "unknown"
        )
        print(f"\n[IBMQuantumVQC] Training on {backend_label}")
        print(
            f"  samples={len(X_train)}  qubits={self.n_qubits}  "
            f"ansatz_params={n_theta}  iterations={max_iterations}"
        )
        print(
            f"  resilience={self.resilience_level}  "
            f"shots={self.default_shots}  DD={self.enable_dd}"
        )

        t0 = time.time()
        COBYLA(maxiter=max_iterations).minimize(objective, initial_theta)
        elapsed = time.time() - t0
        print(
            f"  Done in {elapsed:.1f}s  best_loss={self.best_loss:.4f}"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Classify samples by thresholding <Z0> at zero.
        Paper: "expectation-value measurement on the first qubit, thresholded
                at zero for binary label assignment"

        Returns: np.ndarray of 0/1 labels (0=healthy, 1=AML)
        """
        if self.optimal_params is None:
            raise RuntimeError("Call train() before predict().")
        expectations = self._batch_expectations(X_test, self.optimal_params)
        return (expectations > 0).astype(int)

    # ------------------------------------------------------------------
    # Hybrid mode — hardware validation of simulator-trained params
    # ------------------------------------------------------------------

    def validate_on_hardware(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        token: str = None,
        instance: str = None,
        backend_name: str = None,
    ) -> float:
        """
        Evaluate already-trained parameters on real IBM hardware.

        Use after training in 'simulator' or 'hybrid' mode to get a
        hardware accuracy reading without re-running full optimization.

        Returns: hardware accuracy (float)
        """
        if self.optimal_params is None:
            raise RuntimeError("Train the model first.")

        print("\n[IBMQuantumVQC] Switching to hardware for final validation...")
        saved_mode = self.mode
        self.mode = "hardware"
        self._setup_hardware_backend(token, instance, backend_name)

        preds = self.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        print(f"  Hardware validation accuracy: {acc:.3f}")

        self.mode = saved_mode
        return acc
