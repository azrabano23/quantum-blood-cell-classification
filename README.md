# Quantum & Energy-Based ML for AML Detection

[![arXiv](https://img.shields.io/badge/arXiv-2601.18710-b31b1b.svg)](https://arxiv.org/abs/2601.18710)
[![tests](https://github.com/azrabano23/quantum-blood-cell-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/azrabano23/quantum-blood-cell-classification/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation for:

> **Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia**
> Azra Bano, Larry S. Liebovitch — Rutgers University & Columbia University
> [arXiv:2601.18710](https://arxiv.org/abs/2601.18710)

This is a feasibility study, and the framing is deliberately conservative. The question is not whether quantum methods *beat* a convolutional network — they do not — but whether two learning paradigms that **never invoke backpropagation** (an energy-based network and a 4-qubit variational circuit) can reach accuracy within a small, characterized margin of a tuned CNN on real clinical images, and do so under data scarcity. They can. The result worth a reader's attention is *data efficiency*, and the methodological contribution is treating reproducibility — especially of the variational circuit — as a first-class object rather than a footnote.

All four methods operate on the same 64×64 inputs and the same train/test protocol, so the comparison is controlled. Everything below matches the published paper; runtimes and accuracies are reproduced verbatim from Table I.

---

## Results

Binary classification (AML vs. healthy), 250 samples per class, averaged over 3 random seeds (std < 2%). Runtimes are Qiskit statevector simulation on a laptop (Intel Core i7, 16 GB) and do **not** represent execution on physical quantum hardware.

| Method | Backprop | Accuracy | Train | Test |
|--------|:--------:|---------:|------:|-----:|
| CNN (classical) | yes | **98.4%** | 745 s | 0.19 s |
| Dense NN (classical) | yes | **92.0%** | 0.47 s | 0.001 s |
| **Equilibrium Propagation** (quantum-inspired) | **no** | **86.4%** | 89.4 s | 0.13 s |
| 4-qubit VQC (quantum, Qiskit sim) | no | *see paper* † | 180 s | 1.0 s |

**Equilibrium Propagation reaches 86.4% — only ~12% below the CNN — with no gradient signal anywhere in training.** It sits between the dense baseline (92.0%) and the variational circuit, which is the interesting place for an energy-based method to land: it shows a local, two-phase learning rule is competitive with gradient descent on real clinical data.

**The data-efficiency story is the headline.** The CNN needs the full 250 samples/class to reach 98.4%; at 50 samples/class its accuracy falls to 92.0%. The quantum and quantum-inspired methods hold roughly constant accuracy across the 50→250 range (Fig. 2 in the paper) — i.e. they compete precisely in the regime where annotated data, not model capacity, is the bottleneck. For rare-disease imaging, where expert labels are the scarce resource, that is the property that matters.

† The VQC accuracy reported in the paper is omitted here intentionally. The variational result is acutely sensitive to software stack (see [Reproducibility](#reproducibility)); rather than print a single number, this repo ships the full multi-seed search that produced it (`run_multi_seed.py`) and the verified accuracies (`results_verified.json`). The reported figure is in [arXiv:2601.18710](https://arxiv.org/abs/2601.18710).

---

## Methods

### Dataset

**AML-Cytomorphology_LMU** (Matek et al., 2019) — 18,365 single-cell blood-smear microscopy images from 200 patients (100 AML, 100 healthy controls), acquired at Munich University Hospital with May-Grünwald-Giemsa staining under 100× oil immersion, each image a centered leukocyte annotated by board-certified hematologists per WHO criteria. Available from TCIA: [10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld).

**Protocol.** Images resized to 64×64 for all four methods (controlled comparison, NISQ-realistic compute). Four balanced subsets — 50, 100, 200, 250 samples/class — with stratified 80/20 train/test splits and fixed seeds. EP and VQC consume 20 engineered scalar features per image; CNN consumes pixels.

**20-D feature vector** (5 families): intensity statistics, GLCM texture (contrast, dissimilarity, homogeneity, energy, correlation), morphology (area, eccentricity, solidity, extent), Sobel edge statistics, and FFT spectral energy.

### Classical baselines

- **CNN** (`classical_cnn.py`) — Conv(32→64→128→256) → FC(512→128→2) on 64×64 grayscale; flips, ±15° rotation, ±20% brightness; AdamW + cosine annealing warm restarts.
- **Dense NN** (`classical_dense_nn.py`) — FC(256→128→2) on the 20-D features; Adam + ReduceLROnPlateau.

### Equilibrium Propagation (`equilibrium_propagation.py`)

A layered energy-based network (20→256→128→64→2, tanh) trained in two phases on the global energy
`E = −Σ Wᵢⱼ sᵢ sⱼ − Σ bᵢ sᵢ + ½ Σ sᵢ²`. In the **free phase** the network relaxes to an equilibrium `s⁰` with no supervision; in the **nudged phase** the output is pulled toward the target with strength β = 0.1, reaching `sᵝ`. Each weight then updates from the *local* contrast between the two equilibria,

```
ΔWᵢⱼ ∝ (1/β)( sᵢᵝ sⱼᵝ − sᵢ⁰ sⱼ⁰ )
```

with momentum SGD (μ = 0.9), cosine LR, and early stopping (patience 15). No backpropagation, no global gradient — the update is computable from quantities each unit can observe locally, which is what makes EP relevant to neuromorphic and physical relaxation-based hardware.

### Variational Quantum Circuit (`vqc_classifier.py`)

4 qubits, classically simulated. The 20-D features are reduced to 4 via PCA (≈95% variance) and rescaled to [0, 2π] for the rotation gates. Encoding uses a **ZZFeatureMap** (full entanglement via second-order Pauli-Z terms); the ansatz is a shallow hardware-efficient **RealAmplitudes** block. Training is the standard hybrid loop: execute the circuit, measure ⟨Z₀⟩, compute MSE against labels {−1, +1} classically, and let **COBYLA** (gradient-free, 200 iterations) propose the next parameters — the circuit is treated as a black box, sidestepping the wavefunction-collapse problem that makes in-circuit backprop impossible. Classification thresholds ⟨Z₀⟩ at 0.

### Extended methods (exploratory)

Beyond the paper's four models, the repo includes additional quantum classifiers, run on identical splits for fair comparison:

- `quantum_kernel_svm.py` — fidelity kernel `K(x,y) = |⟨ψ(x)|ψ(y)⟩|²` via ZZFeatureMap, vs. classical RBF/poly SVM.
- `quantum_advantage_method.py` — 8-qubit kernel with **Quantum Kernel Alignment** (trainable feature map), probing whether a 256-D Hilbert-space embedding helps in the low-data regime.
- `vqc_data_reuploading.py` — data re-uploading VQC (Pérez-Salinas et al., 2020) for greater expressibility.
- `ibm_quantum_vqc.py`, `run_on_hardware.py` — execution on real IBM QPUs via `qiskit-ibm-runtime` (EstimatorV2, transpile-once-to-ISA, resilience levels, XY4 dynamical decoupling).

These are exploratory: included because the code is real and the comparisons are controlled, not because they establish a clean quantum advantage.

---

## Reproducibility

CNN, Dense NN, and EP reproduce within ±2% across machines and seeds. The variational circuit does not, and the repo treats that as a finding rather than hiding it. Five mechanisms drive VQC drift across software stacks:

1. **Floating-point non-determinism** — IEEE-754 does not mandate bit-identical results across AVX-512 / NEON / Apple MPS; rounding accumulates.
2. **BLAS backend** — the PCA(20→4) step calls LAPACK SVD; MKL vs. OpenBLAS vs. Accelerate yield numerically-equivalent but FP-distinct eigenvectors (sign flips, component reordering) feeding the circuit.
3. **Qiskit 0.39 → 2.x** — the paper used Qiskit 0.39.0; the primitives rewrite (`StatevectorEstimator`) changed feature-map parameter ordering and expectation computation. This is the single largest source of VQC variation.
4. **COBYLA on a non-convex landscape** — a gradient-free optimizer; small FP differences change simplex moves and the local minimum reached.
5. **EP early stopping** — `patience=15` makes the final weights depend on the exact per-sample update order.

Rather than report one seed, `run_multi_seed.py` performs a 174-combination seed/iteration sweep; `results_verified.json` records the accuracies that hold up under it.

```bash
python3.10 -m venv qml-env && source qml-env/bin/activate
pip install -r requirements.txt   # exact pinned versions
```

---

## Usage

```bash
python run_paper_exact.py /path/to/PKG-AML-Cytomorphology_LMU   # all four methods, paper settings
python run_multi_seed.py  /path/to/dataset                      # VQC seed/iteration sweep
python run_quantum_advantage.py /path/to/dataset                # extended methods + scaling curves
python ibm_quantum_vqc.py /path/to/dataset --mode hardware      # VQC on a real IBM QPU
pytest tests/ -q                                                # shape & invariant tests
```

---

## Repository layout

```
classical_cnn.py            CNN baseline
classical_dense_nn.py       Dense NN on engineered features
equilibrium_propagation.py  EP — two-phase, no backprop
vqc_classifier.py           4-qubit VQC (paper configuration)
run_paper_exact.py          run all four at paper settings
run_multi_seed.py           VQC seed/iteration sweep (174 combinations)
quantum_kernel_svm.py       fidelity-kernel SVM           (extended)
quantum_advantage_method.py 8-qubit trainable kernel / QKA (extended)
vqc_data_reuploading.py     data re-uploading VQC          (extended)
ibm_quantum_vqc.py          IBM hardware execution
run_on_hardware.py          all methods, VQC optionally on QPU
results_verified.json       multi-seed verified accuracies
results_hardware.json       wall-clock + accuracy from hardware runs
tests/                      pytest shape & invariant tests
```

---

## Citation

```bibtex
@article{bano2026qml_aml,
  title   = {Analyzing Images of Blood Cells with Quantum Machine Learning Methods:
             Equilibrium Propagation and Variational Quantum Circuits to Detect
             Acute Myeloid Leukemia},
  author  = {Bano, Azra and Liebovitch, Larry S.},
  journal = {arXiv preprint arXiv:2601.18710},
  year    = {2026}
}
```

**Dataset.** Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). *A single-cell morphological dataset of leukocytes from AML patients and non-malignant controls.* The Cancer Imaging Archive. doi:10.7937/tcia.2019.36f5o9ld

---

## References

- Matek, C. et al. (2019). *A single-cell morphological dataset of leukocytes from AML patients and non-malignant controls.* TCIA. doi:10.7937/tcia.2019.36f5o9ld
- Matek, C. et al. (2021). *Highly accurate differentiation of bone marrow cell morphologies using deep neural networks on a large image dataset.* Blood, 138(20), 1917–1927.
- Scellier, B. & Bengio, Y. (2017). *Equilibrium propagation: bridging the gap between energy-based models and backpropagation.* Frontiers in Computational Neuroscience, 11:24.
- Farhi, E. & Neven, H. (2018). *Classification with quantum neural networks on near-term processors.* arXiv:1802.06002.
- Kandala, A. et al. (2017). *Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets.* Nature, 549, 242–246.
- Pérez-Salinas, A. et al. (2020). *Data re-uploading for a universal quantum classifier.* Quantum, 4, 226.
