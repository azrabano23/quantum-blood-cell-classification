#!/usr/bin/env python3
"""
Quantum Methods Comparison: VQC vs Equilibrium Propagation
=========================================================

This script runs both Variational Quantum Classifier (VQC) and 
Equilibrium Propagation (EP) on the AML-Cytomorphology_LMU dataset
and compares their performance.

Author: A. Zrabano
Research Group: Dr. Liebovitch Lab
Focus: Quantum-Enhanced Medical Image Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
import glob
import os
import time
import pandas as pd

# Import our quantum methods
from src.quantum_networks.ising_classifier import quantum_ising_classifier
from src.quantum_networks.equilibrium_propagation import QuantumEquilibriumPropagation

# Import existing VQC implementation
import sys
sys.path.append('.')
from quantum_demo_complete import QuantumBloodCellClassifier

class AMLDataProcessor:
    """
    Data processor for AML-Cytomorphology_LMU dataset
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.cell_types = {
            # Healthy cells
            'LYT': 0,  # Lymphocytes
            'MON': 0,  # Monocytes  
            'NGS': 0,  # Neutrophil Segmented
            'NGB': 0,  # Neutrophil Band
            
            # AML/Malignant cells
            'MYB': 1,  # Myeloblast
            'MOB': 1,  # Monoblast
            'MMZ': 1,  # Metamyelocyte
            'KSC': 1,  # KSC
            'BAS': 1,  # Basophil
            'EBO': 1,  # EBO
            'EOS': 1,  # Eosinophil
            'LYA': 1,  # LYA
            'MYO': 1,  # Myelocyte
            'PMB': 1,  # Promyelocyte
            'PMO': 1   # Promonocyte
        }
    
    def load_images(self, max_samples_per_class=50):
        """
        Load and preprocess images from the AML dataset
        
        Args:
            max_samples_per_class: Maximum number of samples per cell type
            
        Returns:
            X: Image data (normalized)
            y: Labels (0 for healthy, 1 for AML)
        """
        print(f"📁 Loading AML-Cytomorphology_LMU dataset from: {self.dataset_path}")
        
        X = []
        y = []
        cell_type_counts = {}
        
        for cell_type, label in self.cell_types.items():
            cell_path = os.path.join(self.dataset_path, cell_type)
            
            if not os.path.exists(cell_path):
                print(f"   ⚠️  Cell type {cell_type} not found, skipping...")
                continue
            
            # Get image files
            image_files = glob.glob(os.path.join(cell_path, "*.tiff"))
            image_files.extend(glob.glob(os.path.join(cell_path, "*.tif")))
            
            # Limit samples per class
            if len(image_files) > max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            print(f"   📊 Loading {len(image_files)} {cell_type} images (label: {label})")
            
            for img_file in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize to 8x8 for quantum processing
                    img_resized = cv2.resize(img, (8, 8))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    
                    X.append(img_normalized.flatten())
                    y.append(label)
                    
                except Exception as e:
                    print(f"   ⚠️  Error loading {img_file}: {e}")
                    continue
            
            cell_type_counts[cell_type] = len(image_files)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"   ✅ Loaded {len(X)} total images")
        print(f"   📈 Class distribution: {np.bincount(y)}")
        print(f"   📊 Cell type breakdown: {cell_type_counts}")
        
        return X, y

class QuantumMethodsComparison:
    """
    Compare Variational Quantum Classifier and Equilibrium Propagation
    """
    
    def __init__(self, n_qubits=8, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.results = {}
        
    def run_vqc(self, X_train, y_train, X_test, y_test):
        """
        Run Variational Quantum Classifier
        """
        print(f"\n🔬 Running Variational Quantum Classifier (VQC)")
        print(f"   Architecture: {self.n_qubits} qubits, {self.n_layers} layers")
        
        start_time = time.time()
        
        # Initialize VQC
        vqc = QuantumBloodCellClassifier(n_qubits=self.n_qubits, n_layers=self.n_layers)
        
        # Train VQC
        vqc.train(X_train, y_train, n_epochs=30)
        
        # Make predictions
        y_pred_vqc, quantum_outputs = vqc.predict(X_test)
        
        # Ensure predictions are the right shape
        if len(y_pred_vqc) != len(y_test):
            print(f"   ⚠️  Prediction shape mismatch: {len(y_pred_vqc)} vs {len(y_test)}")
            # Take only the first len(y_test) predictions
            y_pred_vqc = y_pred_vqc[:len(y_test)]
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_vqc)
        
        vqc_results = {
            'method': 'Variational Quantum Classifier',
            'accuracy': accuracy,
            'predictions': y_pred_vqc,
            'training_time': training_time,
            'epochs': len(vqc.training_history),
            'training_history': vqc.training_history,
            'classification_report': classification_report(y_test, y_pred_vqc)
        }
        
        print(f"   ✅ VQC Results:")
        print(f"      Accuracy: {accuracy:.3f}")
        print(f"      Training Time: {training_time:.2f}s")
        print(f"      Epochs: {len(vqc.training_history)}")
        
        return vqc_results
    
    def run_equilibrium_propagation(self, X_train, y_train, X_test, y_test):
        """
        Run Equilibrium Propagation
        """
        print(f"\n⚛️  Running Equilibrium Propagation (EP)")
        print(f"   Architecture: {self.n_qubits} qubits, {self.n_layers} layers")
        
        start_time = time.time()
        
        # Initialize EP
        ep = QuantumEquilibriumPropagation(n_qubits=self.n_qubits, n_layers=self.n_layers)
        
        # Train EP
        ep.train(X_train, y_train, n_epochs=30, learning_rate=0.1)
        
        # Make predictions
        y_pred_ep = ep.predict(X_test)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_ep)
        
        ep_results = {
            'method': 'Equilibrium Propagation',
            'accuracy': accuracy,
            'predictions': y_pred_ep,
            'training_time': training_time,
            'epochs': len(ep.training_history),
            'training_history': ep.training_history,
            'classification_report': classification_report(y_test, y_pred_ep)
        }
        
        print(f"   ✅ EP Results:")
        print(f"      Accuracy: {accuracy:.3f}")
        print(f"      Training Time: {training_time:.2f}s")
        print(f"      Epochs: {len(ep.training_history)}")
        
        return ep_results
    
    def compare_methods(self, vqc_results, ep_results):
        """
        Compare the two quantum methods
        """
        print(f"\n📊 COMPARISON: VQC vs Equilibrium Propagation")
        print(f"=" * 60)
        
        comparison_data = {
            'Method': ['VQC', 'Equilibrium Propagation'],
            'Accuracy': [vqc_results['accuracy'], ep_results['accuracy']],
            'Training Time (s)': [vqc_results['training_time'], ep_results['training_time']],
            'Epochs': [vqc_results['epochs'], ep_results['epochs']]
        }
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Determine winner
        if vqc_results['accuracy'] > ep_results['accuracy']:
            winner = "VQC"
            accuracy_diff = vqc_results['accuracy'] - ep_results['accuracy']
        else:
            winner = "Equilibrium Propagation"
            accuracy_diff = ep_results['accuracy'] - vqc_results['accuracy']
        
        print(f"\n🏆 Winner: {winner} (by {accuracy_diff:.3f} accuracy)")
        
        # Speed comparison
        if vqc_results['training_time'] < ep_results['training_time']:
            speed_winner = "VQC"
            speed_diff = ep_results['training_time'] - vqc_results['training_time']
        else:
            speed_winner = "Equilibrium Propagation"
            speed_diff = vqc_results['training_time'] - ep_results['training_time']
        
        print(f"⚡ Fastest: {speed_winner} (by {speed_diff:.2f}s)")
        
        return df
    
    def plot_comparison(self, vqc_results, ep_results, save_path=None):
        """
        Create comprehensive comparison plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantum Methods Comparison: VQC vs Equilibrium Propagation', fontsize=16)
        
        # 1. Accuracy comparison
        methods = ['VQC', 'Equilibrium Propagation']
        accuracies = [vqc_results['accuracy'], ep_results['accuracy']]
        
        axes[0, 0].bar(methods, accuracies, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Training time comparison
        times = [vqc_results['training_time'], ep_results['training_time']]
        axes[0, 1].bar(methods, times, color=['lightgreen', 'orange'])
        axes[0, 1].set_title('Training Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        for i, v in enumerate(times):
            axes[0, 1].text(i, v + max(times)*0.01, f'{v:.1f}s', ha='center', va='bottom')
        
        # 3. Training convergence
        vqc_epochs = [h['epoch'] for h in vqc_results['training_history']]
        vqc_accs = [h['accuracy'] for h in vqc_results['training_history']]
        ep_epochs = [h['epoch'] for h in ep_results['training_history']]
        ep_accs = [h['accuracy'] for h in ep_results['training_history']]
        
        axes[0, 2].plot(vqc_epochs, vqc_accs, 'b-', label='VQC', linewidth=2)
        axes[0, 2].plot(ep_epochs, ep_accs, 'r-', label='Equilibrium Propagation', linewidth=2)
        axes[0, 2].set_title('Training Convergence')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Training Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confusion matrices
        from sklearn.metrics import confusion_matrix
        
        # VQC confusion matrix
        cm_vqc = confusion_matrix(y_test, vqc_results['predictions'])
        sns.heatmap(cm_vqc, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('VQC Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # EP confusion matrix
        cm_ep = confusion_matrix(y_test, ep_results['predictions'])
        sns.heatmap(cm_ep, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1])
        axes[1, 1].set_title('Equilibrium Propagation Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # 5. Method comparison table
        axes[1, 2].axis('off')
        comparison_text = f"""
Method Comparison Summary:

VQC:
• Accuracy: {vqc_results['accuracy']:.3f}
• Training Time: {vqc_results['training_time']:.1f}s
• Epochs: {vqc_results['epochs']}

Equilibrium Propagation:
• Accuracy: {ep_results['accuracy']:.3f}
• Training Time: {ep_results['training_time']:.1f}s
• Epochs: {ep_results['epochs']}

Key Differences:
• VQC uses gradient descent optimization
• EP uses equilibrium state perturbations
• EP leverages energy-based learning
• VQC is more traditional variational approach
        """
        axes[1, 2].text(0.1, 0.9, comparison_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved to {save_path}")
        
        plt.show()

def main():
    """
    Main function to run the comparison
    """
    print("🧬 Quantum Methods Comparison on AML-Cytomorphology_LMU Dataset")
    print("=" * 70)
    
    # Dataset path
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the AML-Cytomorphology_LMU dataset is available.")
        return
    
    # Load and preprocess data
    processor = AMLDataProcessor(dataset_path)
    X, y = processor.load_images(max_samples_per_class=30)  # Limit for faster processing
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    print(f"   Training class distribution: {np.bincount(y_train)}")
    print(f"   Test class distribution: {np.bincount(y_test)}")
    
    # Initialize comparison
    comparison = QuantumMethodsComparison(n_qubits=8, n_layers=3)
    
    # Run both methods
    vqc_results = comparison.run_vqc(X_train, y_train, X_test, y_test)
    ep_results = comparison.run_equilibrium_propagation(X_train, y_train, X_test, y_test)
    
    # Compare methods
    comparison_df = comparison.compare_methods(vqc_results, ep_results)
    
    # Create visualization
    comparison.plot_comparison(vqc_results, ep_results, 
                             save_path="quantum_methods_comparison_aml.png")
    
    # Save results
    results_summary = {
        'dataset': 'AML-Cytomorphology_LMU',
        'vqc_results': vqc_results,
        'ep_results': ep_results,
        'comparison': comparison_df.to_dict()
    }
    
    print(f"\n💾 Results saved to quantum_methods_comparison_aml.png")
    print(f"🎯 Analysis complete!")

if __name__ == "__main__":
    main()
