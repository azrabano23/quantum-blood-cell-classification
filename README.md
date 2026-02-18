# Quantum Blood Cell Classification

Comparing classical and quantum machine learning methods for Acute Myeloid Leukemia (AML) detection from blood cell microscopy images.

**Paper**: [arXiv:2601.18710](https://arxiv.org/abs/2601.18710)

---

## Methods (Ranked by Performance)

1. **CNN** - Convolutional Neural Network (best)
2. **EP** - Equilibrium Propagation (quantum-inspired)
3. **VQC** - Variational Quantum Classifier
4. **Dense NN** - Dense Neural Network

---

## Requirements

**Python 3.10+**

### Exact Tested Versions
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

### Installation
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

---

## Running Experiments

### Quick Start (All Models)
```bash
python run_verified_experiments.py
```

This runs all 4 models with 250 samples per class and outputs a comparison.

### Individual Models
```bash
python classical_cnn.py           # CNN
python classical_dense_nn.py      # Dense NN  
python equilibrium_propagation.py # Equilibrium Propagation
python vqc_classifier.py          # Quantum Classifier
```

### IBM Quantum Hardware
```bash
python run_on_ibm_quantum.py --setup     # Save API credentials
python run_on_ibm_quantum.py --samples 25  # Run on real hardware
```

---

## Files

| File | Description |
|------|-------------|
| `run_verified_experiments.py` | Runs all 4 models and compares results |
| `classical_cnn.py` | CNN with data augmentation |
| `classical_dense_nn.py` | Dense NN with 20 engineered features |
| `equilibrium_propagation.py` | EP network (no backpropagation) |
| `vqc_classifier.py` | 4-qubit quantum kernel classifier |
| `run_on_ibm_quantum.py` | Run VQC on IBM Quantum hardware |

---

## References

**Dataset**: Matek et al. (2019). A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls. TCIA. [DOI: 10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)

**Equilibrium Propagation**: Scellier & Bengio (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in Computational Neuroscience*.

**VQC**: Farhi & Neven (2018). Classification with quantum neural networks on near term processors. *arXiv:1802.06002*.

---

## License

MIT License

Dataset: CC BY 3.0 ([TCIA](https://www.cancerimagingarchive.net/))
