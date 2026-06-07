"""
Circuit-construction and observable tests for the 4-qubit VQC.

Skipped automatically if Qiskit isn't installed, so the EP tests still run
in a bare environment.
"""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from vqc_classifier import VQCClassifier


def test_classifier_uses_four_qubits_and_matching_pca():
    clf = VQCClassifier(n_qubits=4, n_features=20)
    assert clf.n_qubits == 4
    assert clf.pca.n_components == 4


def test_circuit_has_expected_qubit_count():
    clf = VQCClassifier(n_qubits=4)
    x = np.full(4, np.pi)            # 4 encoded features
    params = np.zeros(12)            # RealAmplitudes(4q, reps=2) -> 12 params
    qc = clf._build_circuit(x, params)
    assert qc.num_qubits == 4


def test_expectation_value_is_a_valid_observable():
    # <Z0> must lie in [-1, 1] for any normalised state.
    clf = VQCClassifier(n_qubits=4)
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 2 * np.pi, size=4)
    params = rng.uniform(-np.pi, np.pi, size=12)
    ev = float(np.asarray(clf._compute_expectation(x, params)).ravel()[0])
    assert -1.0 - 1e-9 <= ev <= 1.0 + 1e-9
