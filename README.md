# Quantum Blood Cell Classification — Paper-Exact Methods

Code for the paper:

**Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia**
Azra Bano, Larry S. Liebovitch — [arXiv:2601.18710](https://arxiv.org/abs/2601.18710)

This branch implements all four methods using parameters stated in the paper. Parameters are not tuned. Where results differ from the paper, discrepancies are documented below. For tuned implementations, see `optimized-methods`.

---

## Paper-Reported Accuracies

| Model | Method | Paper Accuracy |
|-------|--------|---------------|
| **CNN** | Conv32→Conv64→Conv128→Conv256, Adam, data augmentation | 98.4% |
| **Dense NN** | 128→64→32 FC layers on 20 engineered features | 92.0% |
| **EP** | Equilibrium Propagation, NO backprop, beta=0.1, tanh, bidirectional relaxation | 86.4% |
| **VQC** | Qiskit ZZFeatureMap + RealAmplitudes + COBYLA, 4 qubits | 83.0% |

All results from Qiskit statevector simulation (Intel Core i7, 16GB RAM), 250 samples/class, 80/20 stratified split, 3 random seeds (std < 2%). Simulation times do not reflect real quantum hardware.

---

## Known Discrepancies from Paper

Running these implementations with strictly paper-stated parameters does not fully reproduce all reported results:

| Method | Paper Accuracy | Achieved (paper-exact params) | Notes |
|--------|---------------|-------------------------------|-------|
| CNN | 98.4% | ~98% | Architecture not fully specified in paper |
| Dense NN | 92.0% | ~87% | Architecture not specified; 128->64->32 used |
| EP | 86.4% | ~86% | Reproduces well |
| VQC | 83.0% | ~75% | COBYLA plateaus at loss~0.8682 by iter ~80; remaining 120 iters do not improve classification |

**VQC parameter count**: The paper states 8 trainable parameters. `RealAmplitudes(reps=2)` on 4 qubits yields 12 parameters (4 x (reps+1) = 12). The paper may have used reps=1, which gives 8. This implementation uses reps=2 (12 parameters) as written in the paper's circuit description.

See `run_paper_exact.py` for detailed per-run output and discrepancy tracking.

---

## Method Details

### Equilibrium Propagation (EP)
- Architecture: 20 → 256 → 128 → 64 → 2
- Activation: tanh throughout
- **Two-phase training (NO backpropagation)**:
  1. Free phase: network relaxes to equilibrium `s*` without supervision
  2. Nudged phase: output nudged toward target with `beta=0.1`
- **Bidirectional relaxation**: each hidden unit receives input from both adjacent layers (forward + backward via transposed weights), enabling the nudge to propagate through all layers and update all weights
- Weight update: `ΔW_ij ∝ (1/β)(s_i^β s_j^β − s_i* s_j*)`
- Optimizer: momentum SGD (μ=0.9), cosine annealing LR, early stopping (patience=15)
- Reference: Scellier & Bengio (2017)

### Variational Quantum Classifier (VQC)
- 4 qubits, 20 features → PCA(4) → [0, 2π] rescaling
- Feature map: `ZZFeatureMap` (2 reps, full entanglement)
- Ansatz: `RealAmplitudes` (2 reps, 12 trainable parameters — paper states 8; see Known Discrepancies)
- Optimizer: COBYLA (gradient-free), 200 iterations (paper-exact)
- Loss: MSE between `<Z₀>` expectation value and target labels `{-1, +1}`
- Classification: `<Z₀> > 0 → AML`, else Healthy
- Simulator: Qiskit `StatevectorEstimator`
- Reference: Farhi & Neven (2018)

### CNN
- Architecture: Conv(32)→BN→Pool → Conv(64)→BN→Pool → Conv(128)→BN→Pool → Conv(256)→BN→Pool → FC(512)→FC(128)→FC(2)
- Input: 64×64 grayscale images
- Data augmentation: horizontal/vertical flips, ±15° rotation, brightness ±20%
- Optimizer: AdamW (lr=0.001, weight_decay=0.01), cosine annealing warm restarts

### Dense NN
- Input: 20 engineered features (intensity, GLCM, morphology, edge, FFT)
- Architecture: FC(128)→BN→Dropout(0.3) → FC(64)→BN→Dropout(0.3) → FC(32) → FC(2)
- Optimizer: Adam (lr=0.001, weight_decay=0.001), StepLR
- Note: paper does not specify architecture; this is a reasonable implementation

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
3. Update `DATASET_PATH` in the script you want to run

Cell type labels used:
- **Healthy**: LYT, MON, NGS, NGB
- **AML**: MYB, MOB, MMZ, KSC, BAS, EBO, EOS, LYA, MYO, PMO

---

## Running Experiments

### All Models (paper-exact comparison)
```bash
python run_verified_experiments.py
```

Runs all 4 models with 250 samples/class and prints a table comparing achieved vs. paper accuracies.

### Individual Models
```bash
python classical_cnn.py           # CNN
python classical_dense_nn.py      # Dense NN
python equilibrium_propagation.py # EP (paper-exact, no backprop)
python vqc_classifier.py          # VQC (Qiskit, paper-exact)
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
| `run_verified_experiments.py` | Runs all 4 paper models, reports vs. paper targets |
| `classical_cnn.py` | CNN with data augmentation |
| `classical_dense_nn.py` | Dense NN on 20 engineered features |
| `equilibrium_propagation.py` | EP network — bidirectional relaxation, no backprop |
| `vqc_classifier.py` | 4-qubit VQC (Qiskit ZZFeatureMap + COBYLA) |
| `vqc_quantum_kernel.py` | Fast alternative: quantum kernel SVM (not paper-exact) |
| `run_on_ibm_quantum.py` | Run VQC on IBM Quantum hardware |

---

## References

**Paper**: Bano, A., & Liebovitch, L. S. Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia. arXiv:2601.18710.

**Dataset**: Matek et al. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls. TCIA. [DOI: 10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)

**Equilibrium Propagation**: Scellier & Bengio (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in Computational Neuroscience*.

**VQC**: Farhi & Neven (2018). Classification with quantum neural networks on near term processors. *arXiv:1802.06002*.

---

## License

MIT License

Dataset: CC BY 3.0 ([TCIA](https://www.cancerimagingarchive.net/))
