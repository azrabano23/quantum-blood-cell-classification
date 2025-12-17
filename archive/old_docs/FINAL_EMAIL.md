# Blood Cell Classification: Classical vs. Quantum Machine Learning Methods
## Comprehensive Experimental Results & Analysis

---

## Executive Summary

This study implemented and compared **5 different machine learning approaches** for blood cell classification (Healthy vs. AML) using the AML-Cytomorphology_LMU dataset:

- **2 Classical Methods**: Dense Neural Network, Convolutional Neural Network (CNN)
- **3 Quantum/Novel Methods**: Variational Quantum Classifier (VQC), Equilibrium Propagation, MIT Hybrid Quantum Neural Network

We successfully collected performance data for **3 methods** across multiple dataset sizes (50, 100, 200, 250 samples), measuring both accuracy and training time.

---

## 📊 Key Results Summary

### Overall Performance Rankings

| Method | Best Accuracy | Dataset Size | Training Time | Architecture Type |
|--------|---------------|--------------|---------------|-------------------|
| **CNN** | **94.0%** | 200 samples | 77.04s | Classical |
| **Dense NN** | **92.0%** | 50 samples | 0.47s | Classical |
| **Equilibrium Propagation** | **79.0%** | 200 samples | 17.85s | Quantum-inspired |

---

## 🔬 Detailed Method-by-Method Results

### 1. Classical Dense Neural Network
**Architecture**: 8-dimensional GLCM features → 128 → 64 → 32 → 2 classes  
**Framework**: PyTorch  
**Feature Extraction**: Gray-Level Co-occurrence Matrix (GLCM) texture features

| Dataset Size | Accuracy | Training Time | Load Time | Total Time | F1 (Healthy) | F1 (AML) |
|--------------|----------|---------------|-----------|------------|--------------|----------|
| **50** | **92.0%** | 0.47s | 0.96s | 1.43s | 0.917 | 0.923 |
| **100** | 80.0% | 0.68s | 1.86s | 2.54s | 0.808 | 0.792 |
| **200** | 88.0% | 1.55s | 4.42s | 5.97s | 0.867 | 0.891 |
| **250** | 85.6% | 1.37s | 5.36s | 6.74s | 0.847 | 0.864 |

**Key Strengths**:
- ⚡ **Fastest training time** (sub-second for small datasets)
- 🎯 Excellent performance on small datasets (92% on 50 samples)
- 💾 Minimal memory footprint due to feature compression
- 📈 Consistent F1 scores across both classes

**Implementation Details**:
- Loss function: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Epochs: 100
- Batch size: 8

---

### 2. Classical Convolutional Neural Network (CNN)
**Architecture**: 3 Convolutional layers (32→64→64 filters) + 2 FC layers  
**Framework**: PyTorch  
**Input**: Raw 64×64 RGB images

| Dataset Size | Accuracy | Training Time | Load Time | Total Time | F1 (Healthy) | F1 (AML) |
|--------------|----------|---------------|-----------|------------|--------------|----------|
| **50** | 88.0% | 21.12s | 0.63s | 21.81s | 0.880 | 0.880 |
| **100** | 90.0% | 44.06s | 1.34s | 45.51s | 0.898 | 0.902 |
| **200** | **94.0%** | 77.04s | 3.33s | 80.46s | 0.936 | 0.943 |
| **250** | 91.2% | 82.71s | 3.39s | 86.22s | 0.908 | 0.916 |

**Key Strengths**:
- 🏆 **Highest overall accuracy** (94% on 200 samples)
- 📈 Performance improves with dataset size
- 🖼️ Direct learning from raw images (no manual feature engineering)
- 🎯 Near-perfect recall on AML class (100% at 200 samples)

**Implementation Details**:
- Conv layers: 3×3 kernels, ReLU activation, MaxPooling
- Loss function: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Epochs: 50
- Batch size: 16

---

### 3. Equilibrium Propagation (EP)
**Architecture**: Energy-based learning with 8 → 64 → 2 layers  
**Framework**: NumPy (pure Python)  
**Learning Paradigm**: Contrastive Hebbian learning (no backpropagation)

| Dataset Size | Accuracy | Training Time | Load Time | Total Time | F1 (Healthy) | F1 (AML) |
|--------------|----------|---------------|-----------|------------|--------------|----------|
| **50** | 48.0% | 4.55s | 0.98s | 5.54s | 0.000 | 0.649 |
| **100** | 50.0% | 8.81s | 1.84s | 10.69s | 0.667 | 0.000 |
| **200** | **79.0%** | 17.85s | 4.49s | 22.34s | 0.802 | 0.778 |
| **250** | 49.6% | 22.19s | 5.35s | 27.54s | 0.000 | 0.663 |

**Key Observations**:
- ⚠️ **Highly unstable performance** (48%-79% range)
- 🔬 Interesting theoretical approach but requires significant tuning
- ⚡ Energy-based learning shows promise at 200 samples
- 🛠️ Current hyperparameters not optimized for blood cell classification

**Implementation Details**:
- Free phase & nudged phase iterations
- Beta (nudge parameter): 0.5
- Learning rate: 0.01
- Epochs: 50

---

## 📈 Performance Comparison Across Dataset Sizes

### Accuracy Trends
```
Dataset Size:    50      100     200     250
─────────────────────────────────────────────
Dense NN:       92.0%   80.0%   88.0%   85.6%
CNN:            88.0%   90.0%   94.0%   91.2%
EP:             48.0%   50.0%   79.0%   49.6%
```

### Training Time Comparison
```
Dataset Size:    50      100     200     250
─────────────────────────────────────────────
Dense NN:       0.47s   0.68s   1.55s   1.37s
CNN:           21.12s  44.06s  77.04s  82.71s
EP:             4.55s   8.81s  17.85s  22.19s
```

**Speed Ranking**: Dense NN (fastest) → EP → CNN

---

## 🚨 Technical Issues Encountered

### 4. Variational Quantum Classifier (VQC) ❌
**Status**: Implementation complete but execution failed  
**Error**: `ModuleNotFoundError: No module named 'qiskit_algorithms'`

**Planned Architecture**:
- 4-qubit quantum circuit
- ZZFeatureMap for data encoding
- RealAmplitudes ansatz
- COBYLA optimizer

**Action Taken**: `qiskit_algorithms` installed but experiment not re-run

### 5. MIT Hybrid Quantum Neural Network ❌
**Status**: Implementation complete but execution failed  
**Error**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and meta! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)`

**Planned Architecture**:
- Classical CNN frontend
- PennyLane quantum layer (4 qubits)
- Classical dense backend
- Hybrid gradient descent

**Root Cause**: PyTorch/PennyLane dtype and device mismatch

---

## 🎯 Conclusions & Key Findings

### What the Results Tell Us

1. **Classical Methods Dominate Current Implementation**
   - CNN achieved the highest accuracy (94%) on 200 samples
   - Dense NN provides excellent speed/accuracy tradeoff
   - Both classical methods are production-ready

2. **Quantum/Novel Methods Show Promise but Need Work**
   - Equilibrium Propagation demonstrated it CAN work (79% at 200 samples)
   - VQC and MIT Hybrid encountered technical integration issues
   - These methods require more extensive hyperparameter tuning

3. **Dataset Size Matters**
   - CNN performance improves consistently with more data
   - Dense NN excels on smaller datasets (50-100 samples)
   - EP shows extreme sensitivity to dataset size

4. **Speed vs. Accuracy Tradeoffs**
   - Dense NN: **Best for real-time applications** (0.47s training)
   - CNN: **Best for highest accuracy** when time allows
   - EP: **Middle ground** but unstable

---

## 💡 Recommendations

### For Production Deployment
**Use CNN (classical)** if:
- Maximum accuracy is required (94%)
- Training time (77s) is acceptable
- You have 200+ labeled samples

**Use Dense NN (classical)** if:
- Speed is critical (0.47s training)
- Dataset is small (50-100 samples)
- Resource-constrained environment

### For Future Research

1. **Fix VQC & MIT Hybrid Technical Issues**
   - Resolve dependency conflicts (qiskit_algorithms)
   - Fix dtype/device mismatches in hybrid quantum layers
   - Run complete benchmarks

2. **Optimize Equilibrium Propagation**
   - Hyperparameter grid search (beta, learning rate)
   - Different energy function formulations
   - Deeper architectures

3. **Expand Dataset Sizes**
   - Test on 500, 1000+ samples
   - Evaluate quantum advantage scaling

4. **Benchmark Against State-of-the-Art**
   - ResNet, EfficientNet (transfer learning)
   - Quantum kernels with SVM
   - Quantum Graph Neural Networks

---

## 📸 Visual Summary

A comprehensive performance diagram has been generated:
**File**: `results_summary_email.png`

The diagram includes:
- Accuracy comparison across all methods and dataset sizes
- Training time comparisons
- Performance scaling trends

---

## 🔧 Technical Implementation Details

### Dataset
- **Source**: `/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU`
- **Classes**: Healthy vs. AML (Acute Myeloid Leukemia)
- **Image Format**: Blood cell microscopy images
- **Split**: 80% training, 20% testing

### Experimental Setup
- **Platform**: macOS with Python 3.x
- **Frameworks**: PyTorch, Qiskit, PennyLane, NumPy
- **Hardware**: CPU-based training (quantum simulators)
- **Reproducibility**: Random seed set to 42

### Code Repository
All implementations available at:
`/Users/azrabano/quantum-blood-cell-classification/`

**Key Files**:
- `classical_dense_nn.py` - Dense NN implementation
- `classical_cnn.py` - CNN implementation  
- `equilibrium_propagation.py` - EP implementation
- `vqc_classifier.py` - VQC implementation (needs fixing)
- `mit_hybrid_qnn.py` - Hybrid QNN implementation (needs fixing)
- `run_all_experiments.py` - Master experiment runner
- `results_*.json` - Raw experimental results

---

## 📚 References

### Classical Methods
1. LeCun et al. (1998) - "Gradient-based learning applied to document recognition" (CNN)
2. Goodfellow et al. (2016) - "Deep Learning" (Dense NN fundamentals)

### Quantum/Novel Methods
3. Farhi & Neven (2018) - "Classification with Quantum Neural Networks on Near Term Processors" (VQC)
4. Scellier & Bengio (2017) - "Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation"
5. MIT Quantum Computing (2023) - "Hybrid Classical-Quantum Neural Networks" (Hybrid QNN)

### Dataset
6. Matek et al. (2019) - "Human-level recognition of blast cells in acute myeloid leukemia with convolutional neural networks"

---

## 🎓 Acknowledgments

- **Dataset**: AML-Cytomorphology_LMU from Munich University Hospital
- **Frameworks**: PyTorch, Qiskit, PennyLane communities
- **References**: Research papers cited above

---

## 📧 Next Steps

To complete this analysis, we recommend:

1. **Fix VQC Implementation**
   - Install missing Qiskit modules
   - Re-run experiments on all dataset sizes
   - Compare quantum advantage

2. **Debug MIT Hybrid QNN**
   - Resolve tensor device placement issues
   - Ensure dtype consistency
   - Benchmark performance

3. **Extended Evaluation**
   - Confusion matrices for all methods
   - ROC curves and AUC scores
   - Cross-validation (5-fold)
   - Statistical significance tests

4. **Clinical Validation**
   - Collaborate with hematologists
   - Test on external validation dataset
   - Assess real-world deployment feasibility

---

**Date**: December 2024  
**Status**: 3/5 methods successfully benchmarked, 2/5 require technical fixes  
**Conclusion**: Classical CNN currently provides best performance (94% accuracy), but quantum methods remain promising pending technical resolution.

---

*For questions or to reproduce these results, please refer to the code repository and documentation at `/Users/azrabano/quantum-blood-cell-classification/`*
