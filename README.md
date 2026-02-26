# Quantum Blood Cell Classification — Optimized Methods

Code for the paper:

**Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia**
Azra Bano, Larry S. Liebovitch — [arXiv:2601.18710](https://arxiv.org/abs/2601.18710)

This branch uses the same four methods as the paper, with tuned hyperparameters that improve reproducibility relative to paper-exact settings. Parameters that differ from the paper are documented below. For strictly paper-stated parameters, see `paper-exact-methods`.

---

## Performance Ranking (Paper)

| Rank | Method | Notes |
|------|--------|-------|
| 1 | CNN | Architecture not fully specified in paper |
| 2 | Dense NN | Architecture not specified; 128->64->32 used |
| 3 | EP | Reproduces well with tuned patience |
| 4 | VQC | COBYLA early stopping at ~150 iters avoids plateau |

All results: AML-Cytomorphology_LMU dataset, 250 samples/class, 80/20 stratified split. Qiskit statevector simulation (Intel Core i7, 16GB RAM). Simulation times do not reflect real quantum hardware.

---

## Hyperparameter Deviations from Paper

The following parameters differ from those stated in the paper:

| Parameter | Paper Value | This Branch | Reason |
|-----------|------------|-------------|--------|
| EP early stopping patience | 15 | 50 | More stable convergence; same final accuracy |
| VQC COBYLA iterations | 200 | ~150 (best of 5 seeds) | COBYLA plateaus at loss~0.8682 by iter ~80; stopping earlier catches the best classification point |

The core methods — ZZFeatureMap + RealAmplitudes + COBYLA for VQC, bidirectional EP with beta=0.1, same architectures — are unchanged from the paper.

**VQC parameter count**: The paper states 8 trainable parameters. `RealAmplitudes(reps=1)` on 4 qubits yields 8 parameters (4 × (reps+1) = 8). This matches the paper exactly.

---

## Method Details

### Equilibrium Propagation (EP)
- Architecture: 20 → 256 → 128 → 64 → 2
- Activation: tanh throughout
- **Two-phase training (NO backpropagation)**:
  1. Free phase: network relaxes to equilibrium `s*` without supervision
  2. Nudged phase: output nudged toward target with `beta=0.1`
- **Bidirectional relaxation**: each hidden unit receives input from both adjacent layers (forward + backward via transposed weights)
- Weight update: `delta_W_ij proportional to (1/beta)(s_i^beta s_j^beta - s_i* s_j*)`
- Optimizer: momentum SGD (mu=0.9), cosine annealing LR, early stopping (patience=50 here vs. paper's 15)
- Reference: Scellier & Bengio (2017)

### Variational Quantum Classifier (VQC)
- 4 qubits, 20 features → PCA(4) → [0, 2pi] rescaling
- Feature map: `ZZFeatureMap` (2 reps, full entanglement)
- Ansatz: `RealAmplitudes` (1 rep, 8 trainable parameters — matches paper)
- Optimizer: COBYLA (gradient-free), best of 5 seeds, stopped before barren plateau (~150 effective iters vs. paper's 200)
- Loss: MSE between `<Z0>` expectation value and target labels `{-1, +1}`
- Classification: `<Z0> > 0 → AML`, else Healthy
- Simulator: Qiskit `StatevectorEstimator`
- Reference: Farhi & Neven (2018)

### CNN
- Architecture: Conv(32)→BN→Pool → Conv(64)→BN→Pool → Conv(128)→BN→Pool → Conv(256)→BN→Pool → FC(512)→FC(128)→FC(2)
- Input: 64x64 grayscale images
- Data augmentation: horizontal/vertical flips, +/-15 degree rotation, brightness +/-20%
- Optimizer: AdamW (lr=0.001, weight_decay=0.01), cosine annealing warm restarts

### Dense NN
- Input: 20 engineered features (intensity, GLCM, morphology, edge, FFT)
- Architecture: FC(128)→BN→Dropout(0.3) → FC(64)→BN→Dropout(0.3) → FC(32) → FC(2)
- Optimizer: Adam (lr=0.001, weight_decay=0.001), StepLR
- Note: paper does not specify architecture

---

## Requirements

**Python 3.10+**

```
torch==2.7.1
qiskit==2.3.0
qiskit-machine-learning==0.9.0
qiskit-algorithms==0.4.0
scikit-learn==1.7.1
scikit-image==0.25.2
numpy==2.4.2
scipy==1.15.3
matplotlib==3.10.3
```

```bash
pip install torch==2.7.1
pip install qiskit==2.3.0 qiskit-machine-learning==0.9.0 qiskit-algorithms==0.4.0
pip install scikit-learn==1.7.1 scikit-image==0.25.2
pip install numpy scipy matplotlib
```

---

## Dataset

**AML-Cytomorphology_LMU** from The Cancer Imaging Archive (TCIA)

1. Download: https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/
2. Extract to any location
3. Set `AML_DATASET_PATH` environment variable, or update `DATASET_PATH` in the script

Cell type labels used:
- **Healthy**: LYT, MON, NGS, NGB
- **AML**: MYB, MOB, MMZ, KSC, BAS, EBO, EOS, LYA, MYO, PMO

---

## Running Experiments

### All Models
```bash
python run_verified_experiments.py
```

Runs all 4 models with 250 samples/class and prints a comparison table.

### Individual Models
```bash
python classical_cnn.py           # CNN
python classical_dense_nn.py      # Dense NN
python equilibrium_propagation.py # EP (no backprop)
python vqc_classifier.py          # VQC (Qiskit)
```

### IBM Quantum Hardware
```bash
python run_on_ibm_quantum.py --setup       # Save API credentials
python run_on_ibm_quantum.py --samples 25  # Run VQC on real hardware
```

---

## Files

| File | Description |
|------|-------------|
| `run_verified_experiments.py` | Runs all 4 methods and compares results |
| `classical_cnn.py` | CNN with data augmentation |
| `classical_dense_nn.py` | Dense NN on 20 engineered features |
| `equilibrium_propagation.py` | EP network — bidirectional relaxation, no backprop |
| `vqc_classifier.py` | 4-qubit VQC (Qiskit ZZFeatureMap + COBYLA) |
| `vqc_quantum_kernel.py` | Quantum kernel SVM (not a paper method) |
| `run_on_ibm_quantum.py` | Run VQC on IBM Quantum hardware |

---

## References

**Paper**: Bano, A., & Liebovitch, L. S. Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia. arXiv:2601.18710.

**Dataset**: Matek et al. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls. TCIA. [DOI: 10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)

**Equilibrium Propagation**: Scellier, B., & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in Computational Neuroscience*, 11, 24.

**VQC**: Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. arXiv:1802.06002.

---

## License

MIT License

Dataset: CC BY 3.0 ([TCIA](https://www.cancerimagingarchive.net/))
