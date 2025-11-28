# Quantum Blood Cell Classification

**82.7% accuracy** classifying healthy vs AML blood cells using quantum machine learning

## Quick Results

| Metric | Performance |
|--------|-------------|
| **Accuracy** | 82.7% |
| **Cancer Detection (AML Recall)** | 86% |
| **Training Time** | 6.5 minutes |
| **Prediction Speed** | 0.16 sec/image |

## What This Does

Quantum machine learning classifier that distinguishes healthy blood cells from AML (Acute Myeloid Leukemia) cancer cells using:
- 8-qubit quantum circuit (256-dimensional state space)
- Hardware-efficient ansatz with COBYLA optimizer
- GLCM texture feature extraction
- 300 expert-labeled blood cell images from Munich University Hospital

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run classifier (requires blood cell dataset)
python improved_quantum_classifier.py

# View results
open improved_quantum_results.png
```

## Key Innovation

**Solved the "barren plateau" problem** common in quantum ML:
- Traditional approach (Adam optimizer): 53.3% accuracy, no learning
- Our approach (COBYLA + hardware-efficient circuit): **82.7% accuracy**, clear learning curve
- **+29.4% improvement** (+55% relative gain)

## Architecture

```
8 qubits → 3 layers → 48 parameters → 256D quantum state space
├─ Data encoding: RY rotations
├─ Entanglement: Circular CNOT gates  
├─ Variational: RY/RZ rotations
└─ Measurement: 4-qubit expectation values
```

**Features:** GLCM texture analysis (contrast, homogeneity, energy) + statistical moments

## Dataset

**AML-Cytomorphology_LMU** from TCIA:
- 18,365 expert-labeled images total
- Used: 300 samples (150 healthy, 150 AML)
- Source: Munich University Hospital (2014-2017)
- Resolution: 100× magnification
- [Download from TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)

**Cell Types:**
- Healthy: LYT (Lymphocytes), MON (Monocytes), NGS (Neutrophils)
- AML: MYB (Myeloblasts), MOB (Monoblasts), MMZ (Metamyelocytes)

## Performance

**Classification:**
```
              Precision  Recall  F1-Score
Healthy         86%      79%      82%
AML (Cancer)    80%      86%      83%
```

**Timing (6.5 min total):**
- Data loading: 60 sec
- Feature extraction (GLCM): 120 sec  
- Quantum training: 180 sec (20 iterations)
- Prediction: 12 sec (75 images)

**Memory:** ~2 MB (lightweight)

## Files

**Main:**
- `improved_quantum_classifier.py` - Run this (82.7% accuracy)
- `improved_quantum_results.png` - Comprehensive results visualization
- `requirements.txt` - Dependencies

**Documentation:**
- `BENCHMARKING_ANALYSIS.md` - Detailed timing and method comparison
- `TECHNICAL_WRITEUP.md` - Complete scientific details
- `QUANTUM_METHODS_EXPLAINED.md` - Educational guide with diagrams
- `METHOD_COMPARISON_TABLE.md` - Quick reference tables
- `COMPLETE_INDEX.md` - Master documentation index

**Other Results:**
- `quantum_analysis_blood_cells.png` - Original method (53% accuracy)
- `quantum_comparison.png` - Side-by-side comparison

## Medical Significance

**Current Status:** Research prototype approaching clinical utility

**Strengths:**
- 86% cancer detection rate (catches most AML cases)
- Fast inference (0.16 sec/image)
- Lightweight (2 MB memory)

**Limitations:**
- 14% false negative rate (needs <10% for primary diagnosis)
- Requires more data (used 300 of 18K available)

**Use Cases:**
- Screening tool to flag suspicious samples
- Second opinion for pathologists
- Research tool for large-scale analysis

## How It Works

1. **Preprocessing:** Blood cell images → GLCM texture features
2. **Quantum Encoding:** 8 features → 8-qubit quantum state
3. **Quantum Processing:** 3 layers of rotations + entanglement
4. **Measurement:** 4-qubit expectation values → classification
5. **Training:** COBYLA optimizer (gradient-free, avoids barren plateaus)

**Key Quantum Concepts:**
- **Superposition:** Process all 256 states in parallel
- **Entanglement:** Model complex feature correlations
- **Ising Model:** Physics-inspired pattern recognition

## Citation

**Dataset:**
```
Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). 
A Single-cell Morphological Dataset of Leukocytes from AML Patients 
and Non-malignant Controls [Data set]. The Cancer Imaging Archive. 
https://doi.org/10.7937/tcia.2019.36f5o9ld
```

## License

MIT License - Dataset: CC BY 3.0

## Author

A. Zrabano - November 2024
