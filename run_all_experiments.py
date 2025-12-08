#!/usr/bin/env python3
"""
Master Experiment Runner - All Methods Comparison
=================================================

Runs all implemented methods on the blood cell classification task:
1. Classical Dense Neural Network
2. Classical CNN
3. Variational Quantum Classifier (VQC)
4. Equilibrium Propagation
5. MIT Hybrid Quantum-Classical Network

Tests each with different dataset sizes: 50, 100, 200, 250 samples per class

Generates comprehensive comparison plots and tables.

Author: A. Zrabano
"""

import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

def run_method(script_name, method_name):
    """Run a single method script"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {method_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            ['python3', script_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {method_name} took too long")
        return False
    except Exception as e:
        print(f"ERROR running {method_name}: {e}")
        return False

def load_results(results_file):
    """Load results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from: {results_file}")
        return {}

def create_comparison_plots(all_results):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive Method Comparison: Classical vs Quantum Approaches', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Define method names and colors
    methods = {
        'Dense NN': ('results_dense_nn.json', '#2E86AB'),
        'CNN': ('results_cnn.json', '#A23B72'),
        'VQC': ('results_vqc.json', '#F18F01'),
        'Equilibrium Prop': ('results_ep.json', '#C73E1D'),
        'MIT Hybrid QNN': ('results_mit_hybrid.json', '#6A994E'),
        'Improved VQC (Existing)': (None, '#BC4B51')  # Your existing method
    }
    
    sample_sizes = [50, 100, 200, 250]
    
    # Load all results
    results_data = {}
    for method_name, (file, color) in methods.items():
        if file and Path(file).exists():
            results_data[method_name] = load_results(file)
        elif method_name == 'Improved VQC (Existing)':
            # Add placeholder for your existing improved method (82.7% at 150 samples)
            results_data[method_name] = {
                '150': {'accuracy': 0.827, 'train_time': 180, 'total_time': 390}
            }
    
    # 1. Accuracy vs Dataset Size
    ax1 = fig.add_subplot(gs[0, :2])
    for method_name, color in [(m, methods[m][1]) for m in results_data.keys()]:
        if method_name == 'Improved VQC (Existing)':
            ax1.plot([150], [0.827], 'o', markersize=12, color=color, 
                    label=method_name, linewidth=2)
        else:
            sizes = []
            accs = []
            for size in sample_sizes:
                if str(size) in results_data[method_name]:
                    sizes.append(size)
                    accs.append(results_data[method_name][str(size)]['accuracy'])
            if sizes:
                ax1.plot(sizes, accs, 'o-', label=method_name, linewidth=2, 
                        markersize=8, color=color)
    
    ax1.set_xlabel('Dataset Size (samples per class)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])
    
    # 2. Training Time vs Dataset Size
    ax2 = fig.add_subplot(gs[0, 2:])
    for method_name, color in [(m, methods[m][1]) for m in results_data.keys()]:
        if method_name == 'Improved VQC (Existing)':
            ax2.plot([150], [180], 'o', markersize=12, color=color, 
                    label=method_name, linewidth=2)
        else:
            sizes = []
            times = []
            for size in sample_sizes:
                if str(size) in results_data[method_name]:
                    sizes.append(size)
                    times.append(results_data[method_name][str(size)]['train_time'])
            if sizes:
                ax2.plot(sizes, times, 'o-', label=method_name, linewidth=2, 
                        markersize=8, color=color)
    
    ax2.set_xlabel('Dataset Size (samples per class)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Accuracy Comparison Bar Chart (at 250 samples)
    ax3 = fig.add_subplot(gs[1, :2])
    method_names = []
    accuracies = []
    colors_list = []
    
    for method_name, (file, color) in methods.items():
        if method_name == 'Improved VQC (Existing)':
            method_names.append('Improved VQC\n(150 samples)')
            accuracies.append(0.827)
            colors_list.append(color)
        elif method_name in results_data and '250' in results_data[method_name]:
            method_names.append(method_name)
            accuracies.append(results_data[method_name]['250']['accuracy'])
            colors_list.append(color)
    
    bars = ax3.bar(range(len(method_names)), accuracies, color=colors_list, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(method_names)))
    ax3.set_xticklabels(method_names, rotation=15, ha='right')
    ax3.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Comparison (250 samples per class)', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. F1-Score Comparison
    ax4 = fig.add_subplot(gs[1, 2:])
    method_names_f1 = []
    f1_healthy = []
    f1_aml = []
    
    for method_name in results_data.keys():
        if method_name != 'Improved VQC (Existing)' and '250' in results_data[method_name]:
            method_names_f1.append(method_name)
            f1_healthy.append(results_data[method_name]['250']['f1_healthy'])
            f1_aml.append(results_data[method_name]['250']['f1_aml'])
    
    x = np.arange(len(method_names_f1))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, f1_healthy, width, label='Healthy', color='skyblue', edgecolor='black')
    bars2 = ax4.bar(x + width/2, f1_aml, width, label='AML', color='salmon', edgecolor='black')
    
    ax4.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax4.set_title('Per-Class F1-Score (250 samples)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(method_names_f1, rotation=15, ha='right')
    ax4.legend(fontsize=11)
    ax4.set_ylim([0, 1.0])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Total Time Breakdown (Stacked Bar)
    ax5 = fig.add_subplot(gs[2, :2])
    method_names_time = []
    load_times = []
    train_times = []
    pred_times = []
    
    for method_name in results_data.keys():
        if method_name != 'Improved VQC (Existing)' and '250' in results_data[method_name]:
            method_names_time.append(method_name)
            load_times.append(results_data[method_name]['250'].get('load_time', 0))
            train_times.append(results_data[method_name]['250']['train_time'])
            pred_times.append(results_data[method_name]['250'].get('prediction_time', 0))
    
    x = np.arange(len(method_names_time))
    
    p1 = ax5.bar(x, load_times, label='Data Loading', color='#8ECAE6')
    p2 = ax5.bar(x, train_times, bottom=load_times, label='Training', color='#219EBC')
    p3 = ax5.bar(x, pred_times, bottom=np.array(load_times)+np.array(train_times), 
                label='Prediction', color='#023047')
    
    ax5.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax5.set_title('Time Breakdown (250 samples)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(method_names_time, rotation=15, ha='right')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Efficiency: Accuracy per Second of Training
    ax6 = fig.add_subplot(gs[2, 2:])
    method_names_eff = []
    efficiency = []
    colors_eff = []
    
    for method_name, (file, color) in methods.items():
        if method_name != 'Improved VQC (Existing)' and method_name in results_data and '250' in results_data[method_name]:
            method_names_eff.append(method_name)
            acc = results_data[method_name]['250']['accuracy']
            time_taken = results_data[method_name]['250']['train_time']
            efficiency.append(acc / time_taken if time_taken > 0 else 0)
            colors_eff.append(color)
    
    bars = ax6.bar(range(len(method_names_eff)), efficiency, color=colors_eff, alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(method_names_eff)))
    ax6.set_xticklabels(method_names_eff, rotation=15, ha='right')
    ax6.set_ylabel('Accuracy / Training Time (1/sec)', fontsize=12, fontweight='bold')
    ax6.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{eff:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Method Category Comparison
    ax7 = fig.add_subplot(gs[3, :2])
    
    classical_methods = ['Dense NN', 'CNN']
    quantum_methods = ['VQC', 'Equilibrium Prop', 'MIT Hybrid QNN']
    
    classical_accs = []
    quantum_accs = []
    
    for method in classical_methods:
        if method in results_data and '250' in results_data[method]:
            classical_accs.append(results_data[method]['250']['accuracy'])
    
    for method in quantum_methods:
        if method in results_data and '250' in results_data[method]:
            quantum_accs.append(results_data[method]['250']['accuracy'])
    
    categories = ['Classical\nMethods', 'Quantum/Hybrid\nMethods']
    avg_accs = [np.mean(classical_accs) if classical_accs else 0, 
                np.mean(quantum_accs) if quantum_accs else 0]
    std_accs = [np.std(classical_accs) if classical_accs else 0, 
                np.std(quantum_accs) if quantum_accs else 0]
    
    bars = ax7.bar(categories, avg_accs, yerr=std_accs, capsize=10, 
                   color=['#2E86AB', '#F18F01'], alpha=0.7, edgecolor='black')
    ax7.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax7.set_title('Classical vs Quantum/Hybrid Methods (250 samples)', fontsize=14, fontweight='bold')
    ax7.set_ylim([0, 1.0])
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc, std in zip(bars, avg_accs, std_accs):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{acc:.3f} ± {std:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.axis('tight')
    ax8.axis('off')
    
    table_data = []
    headers = ['Method', 'Best Acc', 'Avg Time', 'Category']
    
    for method_name in results_data.keys():
        if method_name == 'Improved VQC (Existing)':
            table_data.append(['Improved VQC*', '82.7%', '180s', 'Quantum'])
        else:
            accs = [results_data[method_name][str(s)]['accuracy'] 
                   for s in sample_sizes if str(s) in results_data[method_name]]
            times = [results_data[method_name][str(s)]['train_time'] 
                    for s in sample_sizes if str(s) in results_data[method_name]]
            
            if accs and times:
                best_acc = max(accs)
                avg_time = np.mean(times)
                category = 'Classical' if method_name in ['Dense NN', 'CNN'] else 'Quantum/Hybrid'
                table_data.append([method_name, f'{best_acc:.1%}', f'{avg_time:.1f}s', category])
    
    table = ax8.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by category
    for i, row in enumerate(table_data):
        color = '#E3F2FD' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
    
    ax8.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('comprehensive_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Comparison plots saved: comprehensive_methods_comparison.png")
    plt.close()

def create_results_table():
    """Create detailed results table"""
    
    methods = {
        'Dense NN': 'results_dense_nn.json',
        'CNN': 'results_cnn.json',
        'VQC': 'results_vqc.json',
        'Equilibrium Prop': 'results_ep.json',
        'MIT Hybrid QNN': 'results_mit_hybrid.json'
    }
    
    sample_sizes = [50, 100, 200, 250]
    
    # Create DataFrame
    rows = []
    for method_name, file in methods.items():
        if Path(file).exists():
            results = load_results(file)
            for size in sample_sizes:
                if str(size) in results:
                    r = results[str(size)]
                    rows.append({
                        'Method': method_name,
                        'Dataset Size': size,
                        'Accuracy': r['accuracy'],
                        'Training Time (s)': r['train_time'],
                        'Total Time (s)': r.get('total_time', 0),
                        'Precision (Healthy)': r['precision_healthy'],
                        'Recall (Healthy)': r['recall_healthy'],
                        'F1 (Healthy)': r['f1_healthy'],
                        'Precision (AML)': r['precision_aml'],
                        'Recall (AML)': r['recall_aml'],
                        'F1 (AML)': r['f1_aml']
                    })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv('detailed_results_table.csv', index=False)
    print("\n✅ Detailed results saved: detailed_results_table.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\nBest Accuracy by Method:")
    best_acc = df.groupby('Method')['Accuracy'].max().sort_values(ascending=False)
    for method, acc in best_acc.items():
        print(f"  {method:25s}: {acc:.3f}")
    
    print("\nAverage Training Time by Method:")
    avg_time = df.groupby('Method')['Training Time (s)'].mean().sort_values()
    for method, time_val in avg_time.items():
        print(f"  {method:25s}: {time_val:.2f}s")
    
    return df

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BLOOD CELL CLASSIFICATION EXPERIMENT")
    print("Comparing Classical and Quantum Machine Learning Methods")
    print("="*80)
    
    start_time = time.time()
    
    # Define all methods
    experiments = [
        ('classical_dense_nn.py', 'Classical Dense Neural Network'),
        ('classical_cnn.py', 'Classical Convolutional Neural Network'),
        ('vqc_classifier.py', 'Variational Quantum Classifier'),
        ('equilibrium_propagation.py', 'Equilibrium Propagation'),
        ('mit_hybrid_qnn.py', 'MIT Hybrid Quantum-Classical Network')
    ]
    
    # Run all experiments
    successful = []
    failed = []
    
    for script, name in experiments:
        if Path(script).exists():
            success = run_method(script, name)
            if success:
                successful.append(name)
            else:
                failed.append(name)
        else:
            print(f"\n⚠️  Script not found: {script}")
            failed.append(name)
    
    # Generate comparison plots and tables
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    create_comparison_plots({})  # Load results inside the function
    df = create_results_table()
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    print(f"\nSuccessful: {len(successful)}/{len(experiments)}")
    for name in successful:
        print(f"  ✅ {name}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(experiments)}")
        for name in failed:
            print(f"  ❌ {name}")
    
    print("\n📊 Output files:")
    print("  • comprehensive_methods_comparison.png")
    print("  • detailed_results_table.csv")
    print("  • results_dense_nn.json")
    print("  • results_cnn.json")
    print("  • results_vqc.json")
    print("  • results_ep.json")
    print("  • results_mit_hybrid.json")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
