#!/usr/bin/env python3
"""
Run Optimized Quantum Experiments
=================================

This script runs both optimized quantum methods:
1. Improved VQC with full entanglement and SPSA optimizer
2. Stabilized Equilibrium Propagation with momentum and adaptive learning

Goal: Demonstrate quantum advantage over classical methods!
"""

import sys
import os

print("="*80)
print("RUNNING OPTIMIZED QUANTUM EXPERIMENTS")
print("="*80)
print()

# Run Optimized Equilibrium Propagation
print("🔬 Starting Optimized Equilibrium Propagation...")
print("-"*80)
try:
    import equilibrium_propagation
    ep_results = equilibrium_propagation.run_experiment(
        "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU",
        sample_sizes=[50, 100, 200, 250]
    )
    print("\n✅ Equilibrium Propagation completed successfully!")
except Exception as e:
    print(f"\n❌ Equilibrium Propagation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print()

# Run Optimized VQC
print("⚛️  Starting Optimized Variational Quantum Classifier...")
print("-"*80)
try:
    import vqc_classifier
    vqc_results = vqc_classifier.run_experiment(
        "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU",
        sample_sizes=[50, 100, 200, 250]
    )
    print("\n✅ VQC completed successfully!")
except Exception as e:
    print(f"\n❌ VQC failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ALL QUANTUM EXPERIMENTS COMPLETED!")
print("="*80)
print("\nCheck results_ep.json and results_vqc.json for detailed results.")
