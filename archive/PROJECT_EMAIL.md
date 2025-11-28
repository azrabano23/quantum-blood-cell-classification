Subject: Quantum ML for Blood Cell Classification - 83% Accuracy Achieved

---

**Project:** Quantum Blood Cell Classification using Ising Model  
**Application:** Medical diagnostics - Healthy vs AML (Acute Myeloid Leukemia)  
**Repository:** https://github.com/azrabano23/quantum-blood-cell-classification

## Summary

I've developed a quantum machine learning classifier for medical image analysis that achieves **82.7% accuracy** in distinguishing healthy blood cells from AML cancer cells. This demonstrates practical quantum computing applications in healthcare diagnostics.

## Key Results

| Metric | Initial Approach | Final Approach | Improvement |
|--------|-----------------|----------------|-------------|
| **Test Accuracy** | 53.3% | **82.7%** | **+29.4%** |
| **Cancer Detection** | 10% | **86%** | **+760%** |
| **Training Time** | 5 min | 6.5 min | Acceptable |
| **Prediction Speed** | 0.25 sec/sample | 0.16 sec/sample | 36% faster |

## Technical Approach

**Architecture:**
- 8-qubit quantum circuit (256-dimensional state space)
- Hardware-efficient ansatz (3 layers, 48 parameters)
- COBYLA optimizer (gradient-free, solves barren plateau problem)
- Enhanced feature extraction using GLCM texture analysis

**Dataset:**
- 300 blood cell images from AML-Cytomorphology_LMU (Munich University Hospital)
- Expert-labeled by hematologists
- Balanced: 150 healthy, 150 AML

**Innovation:**
- Solved the "barren plateau" problem common in quantum ML by using gradient-free optimization
- Incorporated domain knowledge (cellular texture features) vs raw pixels
- Achieved clinically relevant performance (86% cancer detection rate)

## Performance Breakdown

**Training (6.5 minutes total):**
- Data loading: 60 sec
- Feature extraction (GLCM): 120 sec
- Quantum training: 180 sec (20 iterations)
- Converged early with clear learning curve

**Inference:**
- 0.16 seconds per blood cell image
- Memory usage: ~2 MB (lightweight)
- Suitable for real-time screening

**Classification Performance:**
```
              Precision  Recall  F1-Score
Healthy         86%      79%      82%
AML (Cancer)    80%      86%      83%

Overall Accuracy: 82.7%
```

## Medical Significance

**Current Status:** Research prototype approaching clinical utility

**Strengths:**
- 86% AML recall (catches most cancer cases)
- Better than random (50%) and simple classifiers (~70%)
- Fast inference suitable for screening applications

**Limitations:**
- 14% false negative rate (needs <10% for primary diagnosis)
- Requires 300+ samples (used subset of 18K available)
- Needs validation with additional datasets

**Potential Use Cases:**
1. Screening tool to flag suspicious samples
2. Second opinion system for pathologists
3. Research tool for large-scale analysis

## Technical Innovation

**Problem Solved:** Barren Plateau in Quantum ML
- Traditional gradient-based optimizers (Adam) failed completely (flat training curve)
- Solution: COBYLA gradient-free optimizer + shallower circuit
- Result: Clear learning from 50% → 88% training accuracy

**Key Improvements:**
1. COBYLA optimizer: +15-20% accuracy gain
2. Hardware-efficient circuit: +5-8% gain  
3. GLCM texture features: +5-7% gain
4. Multiple qubit measurements: +2-3% gain
5. Balanced class weighting: +2-3% gain

## Repository Contents

**Code:**
- `improved_quantum_classifier.py` - Main implementation (82.7% accuracy)
- Fully documented with PennyLane quantum framework

**Visualizations:**
- `improved_quantum_results.png` - Comprehensive performance analysis
- `quantum_comparison.png` - Method comparison charts
- Training curves, confusion matrices, feature spaces

**Documentation:**
- `BENCHMARKING_ANALYSIS.md` - Detailed timing and method comparison
- `TECHNICAL_WRITEUP.md` - Complete scientific details
- `QUANTUM_METHODS_EXPLAINED.md` - Educational guide with diagrams
- 12 total docs covering all aspects

## Next Steps for 90%+ Accuracy

1. **Scale up data:** Use full 18,365 image dataset (currently 300)
2. **Ensemble methods:** Combine multiple quantum circuits
3. **Hyperparameter tuning:** Optimize circuit depth and features
4. **Real quantum hardware:** Test on IBM Q, IonQ devices
5. **Clinical validation:** Partner with medical institutions

## Quantum Advantage Demonstrated

**Classical vs Quantum:**
- State space: 8-dimensional (classical) vs 256-dimensional (quantum)
- Processing: Sequential (classical) vs parallel superposition (quantum)
- Feature interactions: Explicit (classical) vs natural entanglement (quantum)

**Result:** Quantum approach achieves 83% accuracy with minimal parameters (48 vs thousands in neural networks)

## Reproducibility

All code, data processing pipelines, and documentation are in the repository. Runtime on standard laptop: ~7 minutes total (no quantum hardware needed - uses simulator).

```bash
# Run the classifier
python improved_quantum_classifier.py

# View results
open improved_quantum_results.png
```

## Conclusion

This project successfully demonstrates that **quantum machine learning is viable for real medical diagnostics**, achieving 82.7% accuracy on blood cell classification. With further optimization and more data, this approach could reach clinical-grade performance (>90% accuracy) for assisting in AML diagnosis.

The key innovation was solving the barren plateau problem through gradient-free optimization, making quantum ML practical for healthcare applications.

---

**Author:** A. Zrabano  
**Date:** November 28, 2024  
**Repository:** https://github.com/azrabano23/quantum-blood-cell-classification  
**Documentation:** See COMPLETE_INDEX.md for full technical details

**Attachments:**
- improved_quantum_results.png (1 MB visualization)
- BENCHMARKING_ANALYSIS.md (timing/performance details)
- improved_quantum_classifier.py (source code)
