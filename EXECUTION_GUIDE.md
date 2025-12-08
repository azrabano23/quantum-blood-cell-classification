# Execution Guide: Complete Method Comparison

This guide will help you run all experiments and generate comprehensive comparisons.

## Quick Start (Recommended)

Run everything with a single command:

```bash
python3 run_all_experiments.py
```

This will:
1. ✅ Run all 5 methods (Dense NN, CNN, VQC, Equilibrium Propagation, MIT Hybrid QNN)
2. ✅ Test each with 4 dataset sizes (50, 100, 200, 250 samples per class)
3. ✅ Generate comprehensive comparison plots
4. ✅ Create detailed results tables
5. ✅ Save all metrics to JSON files

**Estimated Time:** 2-4 hours (depending on your hardware)

## Step-by-Step Execution

### Option 1: Run Individual Methods

```bash
# Classical Methods
python3 classical_dense_nn.py      # ~15 min
python3 classical_cnn.py           # ~30 min

# Quantum Methods
python3 vqc_classifier.py          # ~45 min
python3 equilibrium_propagation.py # ~30 min
python3 mit_hybrid_qnn.py          # ~45 min
```

### Option 2: Run by Category

**Classical Methods Only:**
```bash
python3 classical_dense_nn.py
python3 classical_cnn.py
```

**Quantum Methods Only:**
```bash
python3 vqc_classifier.py
python3 equilibrium_propagation.py
python3 mit_hybrid_qnn.py
```

## Output Files

After execution, you'll have:

### Result Files
- `results_dense_nn.json` - Dense NN metrics for all dataset sizes
- `results_cnn.json` - CNN metrics
- `results_vqc.json` - VQC metrics
- `results_ep.json` - Equilibrium Propagation metrics
- `results_mit_hybrid.json` - MIT Hybrid QNN metrics

### Visualization Files
- `comprehensive_methods_comparison.png` - Main comparison dashboard (8 subplots)
- `detailed_results_table.csv` - Excel-compatible results table

### Documentation
- `COMPREHENSIVE_DOCUMENTATION.md` - Complete technical documentation
- This includes:
  - Background of each method
  - Mathematical foundations
  - Implementation details
  - Results analysis
  - Quantum advantage discussion

## Understanding the Results

### Metrics Tracked

For each method and dataset size:
- **Accuracy** - Overall classification accuracy
- **Precision** - Per-class precision (Healthy & AML)
- **Recall** - Per-class recall (Healthy & AML)
- **F1-Score** - Harmonic mean of precision & recall
- **Training Time** - Time to train model
- **Prediction Time** - Time to make predictions
- **Total Time** - Load + Train + Predict

### Visualization Dashboard

The `comprehensive_methods_comparison.png` includes:

1. **Accuracy vs Dataset Size** - How each method scales
2. **Training Time vs Dataset Size** - Computational cost
3. **Accuracy Comparison (250 samples)** - Bar chart of best performance
4. **Per-Class F1-Scores** - Healthy vs AML detection
5. **Time Breakdown** - Stacked bars showing load/train/predict
6. **Training Efficiency** - Accuracy per second
7. **Classical vs Quantum** - Category comparison
8. **Summary Statistics Table** - Quick reference

## Troubleshooting

### Memory Issues

If you run out of memory:

```bash
# Run with smaller dataset sizes
# Edit each script and change:
sample_sizes=[50, 100, 200, 250]
# to:
sample_sizes=[50, 100]
```

### Timeout Issues

For VQC or MIT Hybrid QNN, reduce iterations:

**In vqc_classifier.py:**
```python
# Line 222: Change
train_time = classifier.train(X_train, y_train, max_iterations=50)
# to:
train_time = classifier.train(X_train, y_train, max_iterations=20)
```

**In mit_hybrid_qnn.py:**
```python
# Line 342: Change
train_time = classifier.train(X_train, y_train, epochs=50, batch_size=16)
# to:
train_time = classifier.train(X_train, y_train, epochs=20, batch_size=16)
```

### Missing Dataset

If dataset not found:
```
Error: Dataset not found at: /Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU
```

**Solution:** Update the dataset path in each script or download from:
https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/

## Interpreting Results

### What to Look For

**Classical Methods Performance:**
- Dense NN should be fastest but may plateau earlier
- CNN should achieve higher accuracy with larger datasets

**Quantum Methods Performance:**
- VQC may excel with small datasets (50-100 samples)
- Equilibrium Propagation shows unique energy-based learning
- MIT Hybrid QNN aims for best of both worlds

**Key Comparisons:**
- **Small Data (50-100):** Quantum advantage most visible
- **Large Data (200-250):** Classical methods may dominate
- **Training Time:** Quantum methods typically slower (simulation)
- **Accuracy:** Competitive performance across paradigms

### Expected Ranges

Based on similar experiments:

| Method | Expected Accuracy | Expected Train Time (250 samples) |
|--------|------------------|----------------------------------|
| Dense NN | 75-85% | 10-30 seconds |
| CNN | 80-90% | 60-120 seconds |
| VQC | 70-85% | 180-300 seconds |
| Equilibrium Prop | 70-80% | 30-60 seconds |
| MIT Hybrid QNN | 75-85% | 120-240 seconds |

*Note: Your existing improved VQC achieved 82.7% accuracy*

## Advanced Options

### Custom Dataset Sizes

Edit any script to test different sizes:

```python
# At the bottom of each script, change:
results = run_experiment(dataset_path, sample_sizes=[50, 100, 200, 250])
# to your custom sizes:
results = run_experiment(dataset_path, sample_sizes=[25, 75, 150])
```

### Adjust Hyperparameters

Each script has configurable hyperparameters. Look for:

**Dense NN:**
- `hidden_dims=[128, 64, 32]` - Layer sizes
- `epochs=100` - Training epochs
- `batch_size=16` - Batch size

**CNN:**
- `img_size=64` - Input image size
- `epochs=50` - Training epochs

**VQC:**
- `n_qubits=4` - Number of qubits
- `max_iterations=50` - Optimization steps

**Equilibrium Propagation:**
- `beta=0.5` - Nudging parameter
- `learning_rate=0.01` - Learning rate

**MIT Hybrid QNN:**
- `n_qubits=4` - Quantum layer qubits
- `n_qlayers=2` - Quantum circuit depth

## Next Steps

After running experiments:

1. **Review Results:**
   - Open `comprehensive_methods_comparison.png`
   - Check `detailed_results_table.csv`

2. **Read Documentation:**
   - `COMPREHENSIVE_DOCUMENTATION.md` has full analysis
   - Includes mathematical background and implementation details

3. **Compare to Baseline:**
   - Your existing `improved_quantum_classifier.py` achieved 82.7%
   - See how other methods compare!

4. **Write Your Report:**
   - Use the generated plots and tables
   - Reference the comprehensive documentation
   - Highlight quantum vs classical performance

## Need Help?

**Common Issues:**

1. **Import errors:** Run `pip install -r requirements.txt`
2. **Dataset not found:** Update paths in scripts
3. **Out of memory:** Reduce dataset sizes or batch sizes
4. **Slow execution:** Reduce epochs/iterations
5. **Results differ:** Check random seeds (should be 42)

**Performance Tips:**

- Close other applications to free memory
- Use smaller dataset sizes for quick tests
- Run classical methods first (faster)
- Run quantum methods overnight (slower)

## Summary

You now have a complete pipeline to:
- ✅ Compare 5 different ML methods (2 classical, 3 quantum)
- ✅ Test on multiple dataset sizes
- ✅ Generate publication-quality visualizations
- ✅ Produce comprehensive results tables
- ✅ Document everything with mathematical rigor

**Total Output:**
- 5 JSON result files
- 1 comprehensive visualization
- 1 detailed results table
- Complete technical documentation

Good luck with your experiments! 🚀🔬
