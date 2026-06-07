"""
Shape and invariant tests for the Equilibrium Propagation network.

These don't need the dataset — they exercise the math and the two-phase
relaxation directly, so they run in <1s and catch regressions in the
core learning rule.
"""
import numpy as np
import pytest

from equilibrium_propagation import (
    tanh,
    tanh_derivative,
    EquilibriumPropagationNetwork,
)


def test_tanh_is_clipped_and_bounded():
    # large magnitudes should saturate, not overflow
    assert np.isclose(tanh(1e6), 1.0, atol=1e-6)
    assert np.isclose(tanh(-1e6), -1.0, atol=1e-6)
    assert -1.0 <= tanh(0.3) <= 1.0


def test_tanh_derivative_matches_analytic():
    x = np.linspace(-3, 3, 50)
    assert np.allclose(tanh_derivative(x), 1.0 - np.tanh(x) ** 2)


def test_network_weight_shapes_follow_layer_sizes():
    sizes = [20, 256, 128, 64, 2]
    net = EquilibriumPropagationNetwork(layer_sizes=sizes)
    assert len(net.weights) == len(sizes) - 1
    for i, w in enumerate(net.weights):
        assert w.shape == (sizes[i], sizes[i + 1])


def test_forward_pass_returns_one_state_per_layer():
    sizes = [20, 64, 2]
    net = EquilibriumPropagationNetwork(layer_sizes=sizes)
    x = np.random.randn(20)
    states = net.forward_pass(x, n_iterations=20)
    assert len(states) == len(sizes)
    assert states[-1].shape == (2,)
    # relaxed hidden/output states stay inside the tanh clip range
    assert np.all(np.abs(states[-1]) <= 1.0)


def test_energy_is_a_finite_scalar():
    net = EquilibriumPropagationNetwork(layer_sizes=[20, 64, 2])
    x = np.random.randn(20)
    states = net.forward_pass(x, n_iterations=20)
    e = net.energy(states)
    assert np.isscalar(e) or np.ndim(e) == 0
    assert np.isfinite(e)


def test_nudged_phase_moves_output_toward_target():
    # The defining property of EP: a positive beta should pull the output
    # equilibrium closer to the target than the free-phase equilibrium.
    np.random.seed(0)
    net = EquilibriumPropagationNetwork(layer_sizes=[20, 64, 2])
    x = np.random.randn(20)
    target = np.array([0.9, -0.9])

    free = net.forward_pass(x, n_iterations=80)[-1]
    nudged = net.forward_pass(x, target=target, beta=0.5, n_iterations=80)[-1]

    d_free = np.linalg.norm(free - target)
    d_nudged = np.linalg.norm(nudged - target)
    assert d_nudged <= d_free + 1e-9
