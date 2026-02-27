# Quantum Blood Cell Classification

Code for the paper:

**Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia**
Azra Bano, Larry S. Liebovitch — [arXiv:2601.18710](https://arxiv.org/abs/2601.18710)

---

## Results

| Method | Accuracy | Samples/Class |
|--------|----------|--------------|
| CNN | 98.4% | 250 |
| Dense Neural Network | 92.0% | 250 |
| Equilibrium Propagation | 86.4% | 50 |
| Variational Quantum Classifier | 83.0% | 50 |

Evaluated on the AML-Cytomorphology_LMU dataset using Qiskit statevector simulation.

---

## Methods

### CNN
- Input: 64×64 grayscale images
- Architecture: Conv(32) → Conv(64) → Conv(128) → Conv(256) → FC(512) → FC(128) → FC(2)
- Data augmentation: horizontal/vertical flips, ±15° rotation, brightness ±20%
- Optimizer: AdamW, cosine annealing warm restarts

### Dense Neural Network
- Input: 20 engineered features (intensity, GLCM texture, morphology, edge, FFT)
- Architecture: FC(256) → FC(128) → FC(2)
- Optimizer: Adam, ReduceLROnPlateau

### Equilibrium Propagation
- Input: same 20 engineered features
- Architecture: 20 → 256 → 128 → 64 → 2, tanh activations
- Two-phase training (no backpropagation):
  1. Free phase: network relaxes to equilibrium without supervision
  2. Nudged phase: output nudged toward target with β=0.1
- Weight update: ΔW ∝ (1/β)(s^β s^β − s\* s\*)
- Optimizer: momentum SGD (μ=0.9), cosine annealing LR, early stopping (patience=15)

### Variational Quantum Classifier (VQC)
- 4 qubits
- Feature encoding: 20 features → PCA(4) → rescaled to [0, 2π] → ZZFeatureMap
- Ansatz: RealAmplitudes (2 reps, full entanglement)
- Optimizer: COBYLA, 200 iterations
- Loss: MSE between ⟨Z₀⟩ expectation value and target labels {−1, +1}
- Classification: ⟨Z₀⟩ > 0 → AML, else Healthy

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch, Qiskit, scikit-learn, scikit-image

---

## Dataset

**AML-Cytomorphology_LMU** (Matek et al. 2019) — available from TCIA:
https://doi.org/10.7937/tcia.2019.36f5o9ld

Cell type labels:
- **Healthy**: LYT, MON, NGS, NGB
- **AML**: MYB, MOB, MMZ, KSC, BAS, EBO, EOS, LYA, MYO, PMO

---

## Usage

```bash
python run_paper_exact.py /path/to/PKG-AML-Cytomorphology_LMU
```

Runs all four models and prints a results table vs. paper targets.

Individual models:
```bash
python classical_cnn.py /path/to/dataset
python classical_dense_nn.py /path/to/dataset
python equilibrium_propagation.py /path/to/dataset
python vqc_classifier.py /path/to/dataset
```

---

## Files

| File | Description |
|------|-------------|
| `run_paper_exact.py` | Runs all 4 models with paper-exact parameters |
| `classical_cnn.py` | CNN classifier |
| `classical_dense_nn.py` | Dense NN on engineered features |
| `equilibrium_propagation.py` | EP network — no backpropagation |
| `vqc_classifier.py` | 4-qubit VQC (Qiskit) |
| `requirements.txt` | Python dependencies |

---

## References

Matek, C. et al. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls. TCIA. DOI: 10.7937/tcia.2019.36f5o9ld

Scellier, B., & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in Computational Neuroscience*.

Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. *arXiv:1802.06002*.
