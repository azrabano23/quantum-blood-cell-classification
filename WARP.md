# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This project compares **classical vs quantum machine learning** methods for blood cell classification (Healthy vs AML). It demonstrates that quantum-inspired methods can achieve competitive performance with classical approaches.

**Key Achievement**: Equilibrium Propagation (quantum-inspired) achieves **88% accuracy with ±2% stability**, only 6% behind the classical CNN baseline (94%).

### Core Methods Implemented
1. **Classical CNN** - Best accuracy (94%), requires GPU
2. **Classical Dense NN** - Fastest (<1s), good for small datasets
3. **Equilibrium Propagation** - Quantum-inspired, most stable (±2%), no backpropagation
4. **Variational Quantum Classifier (VQC)** - Pure quantum using Qiskit
5. **MIT Hybrid QNN** - Classical-quantum hybrid architecture

## Essential Commands

### Installation
```bash
# Install all dependencies
pip install qiskit qiskit-machine-learning pennylane torch scikit-image scikit-learn numpy scipy matplotlib

# Or use requirements file
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run all methods and generate comparison (recommended)
python3 run_all_experiments.py

# Run individual methods
python3 equilibrium_propagation.py      # Quantum-inspired (88% accuracy)
python3 classical_cnn.py                 # Classical best (94% accuracy)
python3 classical_dense_nn.py            # Classical fastest (92% accuracy)
python3 vqc_classifier.py                # Pure quantum VQC
python3 mit_hybrid_qnn.py                # Hybrid quantum-classical
python3 improved_quantum_classifier.py   # Original VQC (83% accuracy)

# Run quantum methods only
python3 run_quantum_experiments.py

# Generate visualization of results
python3 generate_email_diagram.py
```

### Dataset Setup
The code expects data at: `/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU`

To use a different location, update `dataset_path` in each script or set the path when running:
- Default path is hardcoded in each script's `if __name__ == "__main__":` section
- Dataset available from [TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)

## Architecture & Design

### Code Organization

**Main Implementations** (root directory):
- Each method is self-contained in a single file
- All follow the same pattern: load data → train → evaluate → save results
- Results saved as JSON files: `results_<method>.json`

**Supporting Files**:
- `src/data_processing/` - Data download utilities (not primary flow)
- `src/quantum_networks/` - Additional quantum implementations
- `archive/` - Historical implementations

### Common Architecture Pattern

Every main classifier follows this structure:
```python
class Classifier:
    def load_data(dataset_folder, max_samples_per_class):
        # Walk directory tree, identify cell types, extract features/images
        # Returns X (features/images), y (labels: 0=healthy, 1=AML)
    
    def train(X_train, y_train):
        # Method-specific training logic
        # Returns training_time
    
    def predict(X_test):
        # Generate predictions
        # Returns predictions, prediction_time
```

### Key Design Principles

1. **Feature Extraction Strategy**:
   - **CNN**: Raw images (64×64 grayscale)
   - **Dense NN, VQC, EP**: GLCM texture features (8 features)
   - GLCM features: contrast, dissimilarity, homogeneity, energy, correlation, ASM, moments, edge statistics

2. **Cell Type Classification**:
   - **Healthy**: LYT (Lymphocytes), MON (Monocytes), NGS (Neutrophils), NGB
   - **AML**: MYB (Myeloblasts), MOB (Monoblasts), MMZ, KSC, BAS, EBO, EOS, LYA, MYO, PMO
   - Binary classification: 0=Healthy, 1=AML

3. **Data Loading Pattern**:
   - All scripts use `os.walk()` to traverse dataset directory
   - Identify cell type from path components
   - Balance classes with `max_samples_per_class` parameter
   - Random seed fixed at 42 for reproducibility

### Equilibrium Propagation (EP) - Critical Implementation Details

EP is the most successful quantum-inspired method. Key optimizations that made it work:

1. **Architecture**: `[8 features → 128 → 64 → 2 classes]`
2. **Xavier Initialization**: Better gradient flow than random init
3. **Two-Phase Training**:
   - Free phase: Network relaxes to equilibrium (50 iterations, beta=0)
   - Nudged phase: Nudge toward target (50 iterations, beta=0.3)
   - Weight update based on correlation difference between phases
4. **Momentum**: 0.9 momentum factor prevents oscillations
5. **Gradient Clipping**: Clips to [-1.0, 1.0] for stability
6. **Adaptive Learning Rate**: Starts at 0.05, decays 20% every 30 epochs
7. **No Backpropagation**: Uses local energy-based learning rules

### Quantum Circuit Design (VQC)

The Variational Quantum Classifier uses:
- **Feature Map**: `ZZFeatureMap` with circular entanglement (2 reps)
- **Ansatz**: `RealAmplitudes` variational form (3 reps)
- **Optimizer**: COBYLA (gradient-free, 100-200 iterations)
- **Qubits**: 4 qubits encoding 4 GLCM features
- **Backend**: Qiskit simulator (no real quantum hardware)

### Result Files

All methods save JSON results to `results_<method>.json` with this schema:
```json
{
  "50": {
    "accuracy": 0.88,
    "train_time": 23.0,
    "predict_time": 0.05,
    "total_time": 25.0,
    "f1_healthy": 0.87,
    "f1_aml": 0.89,
    "precision_healthy": 1.0,
    "precision_aml": 0.8,
    "recall_healthy": 0.77,
    "recall_aml": 1.0
  },
  "100": { ... },
  ...
}
```

## Development Guidelines

### When Modifying Methods

1. **Hyperparameter Locations**:
   - Look in the class `__init__` method for architecture params
   - Look in the `train()` method signature for training params
   - Look at bottom of file in `run_experiment()` for dataset sizes

2. **Adding New Features**:
   - Update `extract_features()` or `load_image()` method
   - Ensure feature count matches model input size
   - Update `n_qubits` for quantum methods if changing feature count

3. **Testing Different Dataset Sizes**:
   - Edit `sample_sizes=[50, 100, 200, 250]` in the script's main block
   - Each size is samples per class (multiply by 2 for total)

### Performance Optimization

**If Training is Too Slow**:
- CNN: Reduce `epochs` (default 50)
- VQC: Reduce `max_iterations` (default 100-200)
- EP: Reduce `n_iterations` in `forward_pass()` (default 50)
- All: Reduce dataset size or use smaller `max_samples_per_class`

**If Memory Issues**:
- CNN: Reduce `batch_size` (default 16)
- CNN: Reduce image size `img_size` (default 64)
- All: Process smaller dataset sizes first

### Expected Performance Ranges

Based on the documented results:

| Method | Accuracy | Training Time (250 samples) | Notes |
|--------|----------|----------------------------|-------|
| Equilibrium Prop | 84-88% | 94-117s | Most stable (±2%) |
| Classical CNN | 88-94% | 77s | Best accuracy, needs GPU |
| Classical Dense NN | 80-92% | <1s | Fastest, good for quick tests |
| VQC | 70-85% | 180-300s | Pure quantum, slower |

### Debugging Tips

1. **Dataset Not Found**: Update the hardcoded path in each script's main block
2. **Import Errors**: Check both old/new Qiskit import patterns (handled with try/except)
3. **Random Results**: Verify `np.random.seed(42)` and framework-specific seeds are set
4. **Poor Accuracy**: 
   - EP: Check if weights initialized with Xavier (not random)
   - EP: Verify momentum and gradient clipping are enabled
   - VQC: Use COBYLA optimizer (not Adam)
   - All: Ensure StandardScaler applied to features

## Workflow Patterns

### Quick Test Workflow
```bash
# Test fastest method first
python3 classical_dense_nn.py

# View results
cat results_dense_nn.json

# If working, test slower methods
python3 equilibrium_propagation.py
python3 classical_cnn.py
```

### Full Comparison Workflow
```bash
# Run everything (takes 2-4 hours)
python3 run_all_experiments.py

# Generate visualization
python3 generate_email_diagram.py

# View comparison plot
open results_summary_email.png

# Check detailed results
ls -lh results_*.json
```

### Modifying a Method Workflow
```bash
# 1. Edit the script (e.g., equilibrium_propagation.py)
# 2. Test on small dataset first
#    - Change sample_sizes=[50] to verify it works
# 3. Run full experiment
#    - Change back to sample_sizes=[50, 100, 200, 250]
# 4. Compare results with previous runs
```

## Important Context

### Medical Dataset Context
- 18,365 total labeled images from Munich University Hospital
- Scripts use subset: 50-250 samples per class for tractability
- Cell images are 100× magnification microscopy
- Expert-labeled by medical professionals
- Privacy note: Dataset is public (TCIA), but follows medical ethics

### Quantum vs Classical Context
- This project demonstrates quantum methods are **competitive** (not superior)
- 88% quantum vs 94% classical = only 6% gap
- Key quantum advantage: Stability (±2% vs ±4%) and no backprop needed
- Future quantum hardware may provide speedup advantages
- EP can run on neuromorphic hardware (energy efficient)

### Publication Context
The project has extensive documentation for academic writing:
- `QUANTUM_SUCCESS.md` - Summary of breakthrough results
- `COMPREHENSIVE_DOCUMENTATION.md` - Full technical details
- `QUANTUM_METHODS_EXPLAINED.md` - Educational explanations
- `TECHNICAL_WRITEUP.md` - Scientific methodology

Use these as reference when explaining methods or results.

## Common Pitfalls

1. **Don't assume test framework exists**: There are no pytest files or test suites. Validation is done through training/test splits in each script.

2. **Don't modify random seeds**: All scripts use `seed=42` for reproducibility. Changing this makes results incomparable to documented performance.

3. **Don't commit result files**: JSON results and PNG plots are gitignored. They are generated artifacts.

4. **Don't use GPU-only features carelessly**: CNN can use CUDA but falls back to CPU. Other methods are CPU-only.

5. **Don't assume dataset is included**: The 18K image dataset is not in the repo (too large). Must be downloaded separately.

## Additional Resources

- Dataset: https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/
- Equilibrium Propagation paper: Scellier & Bengio (2017)
- VQC paper: Farhi & Neven (2018)
- All Python scripts are executable with `#!/usr/bin/env python3`
