#!/usr/bin/env python3
"""
Generate Compact Email-Friendly Summary Diagram
===============================================

Creates a clean, professional diagram suitable for email attachments
showing key results across all methods and dataset sizes.

Run after experiments complete to populate with real results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def load_results(results_file):
    """Load results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def create_email_diagram():
    """Create compact summary diagram for email"""
    
    # Load all results
    methods = {
        'Dense NN': 'results_dense_nn.json',
        'CNN': 'results_cnn.json',
        'VQC': 'results_vqc.json',
        'Equilibrium Prop': 'results_ep.json',
        'MIT Hybrid': 'results_mit_hybrid.json'
    }
    
    sample_sizes = [50, 100, 200, 250]
    
    # Colors for methods
    colors = {
        'Dense NN': '#2E86AB',
        'CNN': '#A23B72',
        'VQC': '#F18F01',
        'Equilibrium Prop': '#C73E1D',
        'MIT Hybrid': '#6A994E'
    }
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    fig.suptitle('Blood Cell Classification: Classical vs Quantum Methods\n' + 
                 'Performance Summary Across Dataset Sizes',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Load all results
    results_data = {}
    for method_name, file in methods.items():
        if Path(file).exists():
            results_data[method_name] = load_results(file)
    
    # If no results yet, create template with placeholders
    if not results_data:
        print("⚠️  No results files found. Creating template diagram.")
        print("Run experiments first, then regenerate this diagram.")
        create_template_diagram(fig, gs, methods, colors, sample_sizes)
        plt.savefig('results_summary_email.png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("✅ Template diagram saved: results_summary_email.png")
        return
    
    # 1. Accuracy Heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    
    accuracy_matrix = []
    method_names = []
    
    for method_name in methods.keys():
        if method_name in results_data:
            method_names.append(method_name)
            row = []
            for size in sample_sizes:
                if str(size) in results_data[method_name]:
                    row.append(results_data[method_name][str(size)]['accuracy'] * 100)
                else:
                    row.append(0)
            accuracy_matrix.append(row)
    
    if accuracy_matrix:
        accuracy_matrix = np.array(accuracy_matrix)
        im = ax1.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=50, vmax=95)
        
        # Add values
        for i in range(len(method_names)):
            for j in range(len(sample_sizes)):
                text = ax1.text(j, i, f'{accuracy_matrix[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax1.set_xticks(range(len(sample_sizes)))
        ax1.set_yticks(range(len(method_names)))
        ax1.set_xticklabels([f'{s}' for s in sample_sizes])
        ax1.set_yticklabels(method_names)
        ax1.set_xlabel('Samples Per Class', fontweight='bold')
        ax1.set_title('Test Accuracy (%)', fontweight='bold', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Accuracy (%)', fontweight='bold')
    
    # 2. Training Time Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    if results_data:
        method_list = []
        time_list = []
        color_list = []
        
        for method_name in methods.keys():
            if method_name in results_data and '250' in results_data[method_name]:
                method_list.append(method_name.replace(' ', '\n'))
                time_list.append(results_data[method_name]['250']['train_time'])
                color_list.append(colors[method_name])
        
        if method_list:
            bars = ax2.barh(range(len(method_list)), time_list, color=color_list, alpha=0.8)
            ax2.set_yticks(range(len(method_list)))
            ax2.set_yticklabels(method_list, fontsize=9)
            ax2.set_xlabel('Time (seconds)', fontweight='bold')
            ax2.set_title('Training Time\n(250 samples)', fontweight='bold', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add values
            for i, (bar, time) in enumerate(zip(bars, time_list)):
                ax2.text(time + max(time_list)*0.02, i, f'{time:.1f}s',
                        va='center', fontweight='bold', fontsize=9)
    
    # 3. Accuracy vs Dataset Size (Line Plot)
    ax3 = fig.add_subplot(gs[1, :])
    
    for method_name in methods.keys():
        if method_name in results_data:
            sizes = []
            accs = []
            for size in sample_sizes:
                if str(size) in results_data[method_name]:
                    sizes.append(size)
                    accs.append(results_data[method_name][str(size)]['accuracy'] * 100)
            
            if sizes:
                ax3.plot(sizes, accs, 'o-', label=method_name, 
                        color=colors[method_name], linewidth=2.5, markersize=10)
    
    ax3.set_xlabel('Dataset Size (samples per class)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=11)
    ax3.set_title('Accuracy Scaling with Dataset Size', fontweight='bold', fontsize=12)
    ax3.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([50, 100])
    
    # 4. Best Performance Summary (250 samples)
    ax4 = fig.add_subplot(gs[2, 0])
    
    if results_data:
        method_list = []
        acc_list = []
        color_list = []
        
        for method_name in methods.keys():
            if method_name in results_data and '250' in results_data[method_name]:
                method_list.append(method_name.replace(' ', '\n'))
                acc_list.append(results_data[method_name]['250']['accuracy'] * 100)
                color_list.append(colors[method_name])
        
        if method_list:
            bars = ax4.bar(range(len(method_list)), acc_list, color=color_list, alpha=0.8, edgecolor='black')
            ax4.set_xticks(range(len(method_list)))
            ax4.set_xticklabels(method_list, fontsize=8, rotation=0)
            ax4.set_ylabel('Accuracy (%)', fontweight='bold')
            ax4.set_title('Best Performance\n(250 samples)', fontweight='bold', fontsize=11)
            ax4.set_ylim([0, 100])
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add values
            for bar, acc in zip(bars, acc_list):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 5. Classical vs Quantum Comparison
    ax5 = fig.add_subplot(gs[2, 1])
    
    if results_data:
        classical = ['Dense NN', 'CNN']
        quantum = ['VQC', 'Equilibrium Prop', 'MIT Hybrid']
        
        classical_accs = []
        quantum_accs = []
        
        for method in classical:
            if method in results_data and '250' in results_data[method]:
                classical_accs.append(results_data[method]['250']['accuracy'] * 100)
        
        for method in quantum:
            if method in results_data and '250' in results_data[method]:
                quantum_accs.append(results_data[method]['250']['accuracy'] * 100)
        
        if classical_accs and quantum_accs:
            categories = ['Classical\nMethods', 'Quantum/Hybrid\nMethods']
            means = [np.mean(classical_accs), np.mean(quantum_accs)]
            stds = [np.std(classical_accs), np.std(quantum_accs)]
            
            bars = ax5.bar(categories, means, yerr=stds, capsize=10,
                          color=['#2E86AB', '#F18F01'], alpha=0.8, edgecolor='black')
            ax5.set_ylabel('Accuracy (%)', fontweight='bold')
            ax5.set_title('Method Category\nComparison', fontweight='bold', fontsize=11)
            ax5.set_ylim([0, 100])
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add values
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                        f'{mean:.1f}%\n±{std:.1f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
    
    # 6. Key Statistics Table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    if results_data:
        table_data = []
        
        for method_name in methods.keys():
            if method_name in results_data:
                # Best accuracy across all sizes
                best_acc = 0
                avg_time = 0
                count = 0
                
                for size in sample_sizes:
                    if str(size) in results_data[method_name]:
                        acc = results_data[method_name][str(size)]['accuracy']
                        best_acc = max(best_acc, acc)
                        avg_time += results_data[method_name][str(size)]['train_time']
                        count += 1
                
                if count > 0:
                    avg_time /= count
                    category = 'Classical' if method_name in ['Dense NN', 'CNN'] else 'Quantum'
                    table_data.append([
                        method_name,
                        f'{best_acc*100:.1f}%',
                        f'{avg_time:.1f}s',
                        category
                    ])
        
        if table_data:
            headers = ['Method', 'Best\nAcc', 'Avg\nTime', 'Type']
            table = ax6.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4A90E2')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color rows
            for i in range(len(table_data)):
                color = '#E3F2FD' if i % 2 == 0 else '#FFFFFF'
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor(color)
    
    ax6.set_title('Performance\nSummary', fontweight='bold', fontsize=11, pad=10)
    
    # Save
    plt.savefig('results_summary_email.png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("\n✅ Email diagram saved: results_summary_email.png")
    plt.close()

def create_template_diagram(fig, gs, methods, colors, sample_sizes):
    """Create template when no results available"""
    
    # Template text
    ax = fig.add_subplot(gs[:, :])
    ax.axis('off')
    
    template_text = """
    📊 RESULTS SUMMARY DIAGRAM (TEMPLATE)
    
    This diagram will display:
    
    1. Accuracy Heatmap
       - All methods × all dataset sizes
       - Color-coded performance
    
    2. Training Time Comparison
       - Horizontal bar chart
       - For 250 samples per class
    
    3. Accuracy vs Dataset Size
       - Line plot showing scaling
       - All methods compared
    
    4. Best Performance (250 samples)
       - Bar chart of final accuracy
    
    5. Classical vs Quantum
       - Average performance comparison
       - With error bars
    
    6. Statistics Table
       - Best accuracy per method
       - Average training time
       - Method category
    
    ⚠️  TO GENERATE WITH REAL DATA:
    
    1. Run experiments:
       python3 run_all_experiments.py
    
    2. Regenerate diagram:
       python3 generate_email_diagram.py
    
    3. Use in email with:
       EMAIL_RECAP.md template
    """
    
    ax.text(0.5, 0.5, template_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', horizontalalignment='center',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating Email Summary Diagram")
    print("="*60)
    
    create_email_diagram()
    
    print("\n📧 Email diagram ready!")
    print("Use with EMAIL_RECAP.md template")
    print("="*60 + "\n")
