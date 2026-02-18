# Quantum Blood Cell Classification
**Comparing Classical and Quantum Machine Learning for Acute Myeloid Leukemia Detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

This project **proves Quantum Machine Learning is competitive** with classical methods for real-world medical imaging. We compare classical, Quantum-inspired, and pure Quantum approaches for automated detection of Acute Myeloid Leukemia (AML) from blood cell microscopy images.

### Methods Implemented

1. **Classical CNN**
   - Architecture: Conv(32) → Conv(64) → Conv(128) → FC(256) → FC(128) → 2 classes
   - Data augmentation: flips, rotation, brightness, zoom
   - Regularization: dropout, weight decay, gradient clipping
   
2. **Classical Dense NN**
   - Architecture: 8 GLCM features → 128 → 64 → 32 → 2 classes
   - Feature extraction: texture analysis (GLCM)

3. **Equilibrium Propagation** (Quantum-Inspired)
   - Architecture: 20 features → 256 → 128 → 64 → 2 classes
   - Energy-based learning (no backpropagation needed)
   - Features: statistical + GLCM + morphology + edge + frequency

4. **Variational Quantum Classifier** (Pure Quantum)
   - 4-qubit quantum circuit
   - ZZFeatureMap encoding + RealAmplitudes ansatz
   - Qiskit implementation

5. **MIT Hybrid QNN** (Quantum-Classical Hybrid)
   - Hybrid architecture combining quantum and classical layers
   - Feature extraction with quantum encoding

---
## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/azrabano23/quantum-blood-cell-classification.git
cd quantum-blood-cell-classification

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=1.9.0
qiskit>=0.36.0
qiskit-machine-learning>=0.4.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
```

### Dataset Setup

1. Download the **AML-Cytomorphology_LMU** dataset from [TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)
2. Extract to: `/path/to/PKG - AML-Cytomorphology_LMU`
3. Update dataset path in scripts (line ~460-465 in each file)

### Running Experiments

```bash
# Run CNN
python classical_cnn.py

# Run Dense NN
python classical_dense_nn.py

# Run Equilibrium Propagation
python equilibrium_propagation.py

# Run Variational Quantum Classifier
python vqc_classifier.py

# Run MIT Hybrid QNN
python mit_hybrid_qnn.py

# Run all experiments
python run_all_experiments.py
```

---

## 📁 Repository Structure

### Core Implementations
```
classical_cnn.py              # CNN with data augmentation
classical_dense_nn.py         # Dense network with GLCM features
equilibrium_propagation.py    # Quantum-inspired EP with 20 features
vqc_classifier.py             # Variational quantum classifier
mit_hybrid_qnn.py             # Hybrid quantum-classical network
final_evaluation.py           # Comprehensive evaluation framework
feature_extractor.py          # 32-feature extraction
```

### Utilities
```
run_all_experiments.py       # Master script to run all methods
run_quantum_experiments.py   # Run quantum methods only
generate_email_diagram.py    # Visualize results
requirements.txt             # Python dependencies
```

### Results
```
results_cnn.json             # CNN experiment results
results_dense_nn.json        # Dense NN results
results_ep.json              # Equilibrium propagation results
results_vqc.json             # VQC results
results_summary_email.png    # Performance visualization
```

### Documentation
```
README.md                    # This file
archive/                     # Archived documentation and legacy code
```
---

## 🔗 References

### Dataset
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). *A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls* [Data set]. The Cancer Imaging Archive. [DOI: 10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)

### Methods
1. **Equilibrium Propagation**: Scellier, B., & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in computational neuroscience*, 11, 24.

2. **Variational Quantum Classifier**: Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. *arXiv preprint arXiv:1802.06002*.

3. **Quantum Machine Learning**: Benedetti, M., Lloyd, E., Sack, S., & Fiorentini, M. (2019). Parameterized quantum circuits as machine learning models. *Quantum Science and Technology*, 4(4), 043001.

4. **CNNs for Medical Imaging**: LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

5. **Data Augmentation**: Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 1-48.

### Quantum Computing Resources
- **Qiskit Documentation**: [qiskit.org/documentation](https://qiskit.org/documentation/)
- **IBM Quantum**: [quantum-computing.ibm.com](https://quantum-computing.ibm.com/)
- **Qiskit Machine Learning**: [qiskit.org/ecosystem/machine-learning](https://qiskit.org/ecosystem/machine-learning/)

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@inproceedings{bano2026quantum,
  author = {Bano, Azra and Liebovitch, Larry S.},
  title = {Analyzing Images of Blood Cells with Quantum Machine Learning Methods: Equilibrium Propagation and Variational Quantum Circuits to Detect Acute Myeloid Leukemia},
  booktitle = {IEEE World Congress on Computational Intelligence (WCCI)},
  year = {2026},
  note = {Special Session on Quantum Machine Learning},
  url = {https://github.com/azrabano23/conference-paper-quantum-ml}
}

@software{bano2024quantum_code,
  author = {Bano, Azra},
  title = {Quantum Blood Cell Classification: Implementation Code},
  year = {2024},
  url = {https://github.com/azrabano23/quantum-blood-cell-classification}
}
```

---

## 📜 License

MIT License - See LICENSE file for details

Dataset: CC BY 3.0 License ([TCIA](https://www.cancerimagingarchive.net/))
---

## Acknowledgments

- Munich University Hospital for the AML-Cytomorphology dataset
- Qiskit and IBM Quantum teams for quantum computing tools
- PyTorch and scikit-learn communities for ML frameworks
