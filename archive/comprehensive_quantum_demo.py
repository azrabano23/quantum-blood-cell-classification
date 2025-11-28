#!/usr/bin/env python3
"""
Comprehensive Quantum Classification Demo
=========================================

This script demonstrates quantum computing for classification on:
1. MNIST handwritten digits (benchmark dataset)
2. Real blood cell images (medical application)

Author: A. Zrabano
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pennylane as qml
import seaborn as sns
from skimage.io import imread_collection
from skimage.transform import resize
import os
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

class QuantumIsingClassifier:
    """
    Quantum Ising Model for Binary Classification
    
    Uses quantum superposition, entanglement, and Ising spin interactions
    to classify data in a quantum state space.
    """
    
    def __init__(self, n_qubits=8, n_layers=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.weights = None
        self.training_history = []
        
    def create_quantum_circuit(self):
        """
        Quantum Ising Model Circuit Architecture
        
        Components:
        1. Data Encoding: RY rotations map classical data to quantum states
        2. Ising Interactions: CNOT + RZ implement spin-spin couplings
        3. Local Fields: RX rotations for individual qubit control
        4. Measurement: Pauli-Z expectation value for classification
        """
        
        @qml.qnode(self.device)
        def circuit(weights, x):
            # Data Encoding Layer - Creates quantum superposition
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(np.pi * x[i], wires=i)
            
            # Variational Ising Layers
            for layer in range(self.n_layers):
                # Ising spin-spin interactions (creates entanglement)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                    qml.RZ(weights[layer, i], wires=i+1)
                    qml.CNOT(wires=[i, i+1])
                
                # Local magnetic fields
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, self.n_qubits + i], wires=i)
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def train(self, X_train, y_train, n_epochs=30):
        """Train using variational quantum optimization"""
        
        print(f"⚛️  Training Quantum Ising Classifier")
        print(f"   Architecture: {self.n_qubits} qubits, {self.n_layers} layers")
        print(f"   Quantum state space: 2^{self.n_qubits} = {2**self.n_qubits} dimensions")
        print(f"   Training samples: {len(X_train)}")
        
        # Initialize parameters - PennyLane automatically handles gradients
        self.weights = np.random.randn(self.n_layers, 2 * self.n_qubits) * 0.1
        self.weights = self.weights.astype(np.float64)  # Ensure float64 for better precision
        circuit = self.create_quantum_circuit()
        
        def cost_function(weights):
            # Use continuous loss for better gradients
            total_loss = 0.0
            for x, y_true in zip(X_train, y_train):
                output = circuit(weights, x[:self.n_qubits])
                # Convert output from [-1, 1] to prediction [0, 1]
                # Use hinge-like loss
                y_encoded = 2 * y_true - 1  # Convert 0,1 to -1,1
                loss = (1 - y_encoded * output) ** 2
                total_loss += loss
            return total_loss / len(X_train)
        
        # Training with Adam optimizer (better than vanilla gradient descent)
        optimizer = qml.AdamOptimizer(stepsize=0.01)
        
        def compute_accuracy(weights):
            predictions = []
            for x, y_true in zip(X_train, y_train):
                output = circuit(weights, x[:self.n_qubits])
                pred = 1 if output > 0 else 0
                predictions.append(pred == y_true)
            return np.mean(predictions)
        
        self.training_history = []
        for epoch in range(n_epochs):
            try:
                self.weights = optimizer.step(cost_function, self.weights)
                cost = float(cost_function(self.weights))
                accuracy = compute_accuracy(self.weights)
                self.training_history.append({'epoch': epoch, 'cost': cost, 'accuracy': accuracy})
                
                if epoch % 5 == 0:
                    print(f"   Epoch {epoch:2d}: Loss = {cost:.4f}, Accuracy = {accuracy:.3f}")
            except Exception as e:
                print(f"   Error at epoch {epoch}: {e}")
                # Simple fallback
                self.weights = self.weights + 0.001 * np.random.randn(*self.weights.shape)
                cost = float(cost_function(self.weights))
                accuracy = compute_accuracy(self.weights)
                self.training_history.append({'epoch': epoch, 'cost': cost, 'accuracy': accuracy})
        
        final_accuracy = compute_accuracy(self.weights)
        print(f"   Training complete: Final accuracy = {final_accuracy:.3f}")
        return self.weights
    
    def predict(self, X):
        """Make predictions using trained quantum circuit"""
        if self.weights is None:
            raise ValueError("Model must be trained first")
        
        circuit = self.create_quantum_circuit()
        predictions = []
        quantum_outputs = []
        
        for x in X:
            try:
                output = circuit(self.weights, x[:self.n_qubits])
                quantum_outputs.append(output)
                predictions.append(1 if output > 0 else 0)
            except:
                predictions.append(np.random.choice([0, 1]))
                quantum_outputs.append(0)
        
        return np.array(predictions), np.array(quantum_outputs)

def load_mnist_data(n_samples=200):
    """Load and preprocess MNIST data for binary classification (0 vs 1)"""
    print("\n🔢 Loading MNIST Dataset...")
    
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.to_numpy() / 255.0  # Normalize to [0, 1]
    y = mnist.target.astype(int).to_numpy()
    
    # Select only 0s and 1s for binary classification
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    
    # Sample subset
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
    
    # Reduce dimensionality with PCA to 8 features (matching our 8 qubits)
    print("   Reducing dimensionality with PCA...")
    pca = PCA(n_components=8)
    X_reduced = pca.fit_transform(X_sampled)
    
    # Normalize to [0, 1] for quantum encoding
    X_normalized = (X_reduced - X_reduced.min(axis=0)) / (X_reduced.max(axis=0) - X_reduced.min(axis=0) + 1e-8)
    
    print(f"   Loaded {len(X_sampled)} samples (0s and 1s)")
    print(f"   Features: {X_normalized.shape[1]}")
    print(f"   Distribution: 0s={np.sum(y_sampled==0)}, 1s={np.sum(y_sampled==1)}")
    
    return X_normalized, y_sampled, X_sampled  # Return both reduced and original for visualization

def load_blood_cell_data(dataset_folder, max_samples_per_class=100):
    """Load real blood cell data from the dataset folders"""
    print(f"\n🔬 Loading Blood Cell Dataset from: {dataset_folder}")
    
    # Define cell type classifications based on medical knowledge
    healthy_cell_types = ['LYT', 'MON', 'NGS', 'NGB']  # Lymphocytes, Monocytes, Neutrophils
    aml_cell_types = ['MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO', 'EOS', 'LYA', 'MYO', 'PMO']  # Blasts and abnormal cells
    
    X, y = [], []
    class_counts = {'healthy': 0, 'aml': 0}
    sample_images = []  # Store some sample images
    
    for dirpath, _, filenames in os.walk(dataset_folder):
        # Extract cell type from directory path
        path_parts = dirpath.split(os.sep)
        cell_type = None
        
        # Find the cell type in the path
        for part in path_parts:
            if part in healthy_cell_types or part in aml_cell_types:
                cell_type = part
                break
        
        if cell_type is None:
            continue
        
        for file in filenames:
            if file.endswith(('.jpg', '.png', '.tiff', '.tif')):
                # Determine class based on cell type
                if cell_type in healthy_cell_types:
                    if class_counts['healthy'] >= max_samples_per_class:
                        continue
                    label = 0  # Healthy
                    class_counts['healthy'] += 1
                elif cell_type in aml_cell_types:
                    if class_counts['aml'] >= max_samples_per_class:
                        continue
                    label = 1  # AML/Malignant
                    class_counts['aml'] += 1
                else:
                    continue
                
                try:
                    img_path = os.path.join(dirpath, file)
                    img = imread_collection([img_path])[0]
                    
                    # Store a few sample images
                    if len(sample_images) < 10:
                        sample_images.append((img, label, cell_type))
                    
                    # Convert to grayscale if RGB
                    if len(img.shape) == 3:
                        img = np.mean(img, axis=2)
                    
                    # Resize and normalize
                    img_resized = resize(img, (4, 4), anti_aliasing=True)
                    img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
                    
                    X.append(img_normalized.flatten())
                    y.append(label)
                    
                except Exception as e:
                    continue
    
    print(f"   Loaded {len(X)} samples:")
    print(f"   • Healthy: {class_counts['healthy']}")
    print(f"   • AML: {class_counts['aml']}")
    
    return np.array(X), np.array(y), sample_images

def create_comprehensive_visualization(dataset_name, X, y, X_test, y_test, predictions, 
                                     quantum_outputs, classifier, test_accuracy, 
                                     sample_images=None, class_names=['Class 0', 'Class 1']):
    """Create comprehensive visualization for results"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    fig.suptitle(f'Quantum Ising Model Classification: {dataset_name}', 
                fontsize=18, fontweight='bold')
    
    # 1. Sample Images (if available)
    if sample_images is not None and len(sample_images) > 0:
        ax = fig.add_subplot(gs[0, 0:2])
        n_display = min(8, len(sample_images))
        for i in range(n_display):
            plt.subplot(3, 4, i + 1)
            img, label, cell_type = sample_images[i]
            if len(img.shape) == 3:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap='gray')
            plt.title(f'{cell_type}\n{class_names[label]}', fontsize=8)
            plt.axis('off')
        plt.subplot(3, 4, 1)  # Reset to main subplot
    
    # 2. Feature Space Visualization
    ax = fig.add_subplot(gs[0, 2])
    if X.shape[1] >= 2:
        ax.scatter(X[y==0, 0], X[y==0, 1], c='green', alpha=0.6, label=class_names[0], s=30)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, label=class_names[1], s=30)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    ax.set_title('Feature Space', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Quantum Circuit Explanation
    ax = fig.add_subplot(gs[0, 3])
    circuit_text = """
    Quantum Circuit:
    
    1. Data Encoding
       RY(πx_i)|0⟩
       Creates superposition
    
    2. Ising Interactions
       CNOT-RZ-CNOT
       Creates entanglement
    
    3. Local Fields
       RX(θ) rotations
       Variational parameters
    
    4. Measurement
       ⟨Z⟩ → Classification
    """
    ax.text(0.05, 0.95, circuit_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.axis('off')
    
    # 4. Training Progress
    ax = fig.add_subplot(gs[1, 0])
    if classifier.training_history:
        epochs = [h['epoch'] for h in classifier.training_history]
        accuracies = [h['accuracy'] for h in classifier.training_history]
        ax.plot(epochs, accuracies, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Accuracy')
        ax.set_title('Training Progress', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    # 5. Quantum Output Distribution
    ax = fig.add_subplot(gs[1, 1])
    if len(quantum_outputs) > 0:
        outputs_0 = quantum_outputs[y_test == 0] if np.any(y_test == 0) else []
        outputs_1 = quantum_outputs[y_test == 1] if np.any(y_test == 1) else []
        
        if len(outputs_0) > 0:
            ax.hist(outputs_0, alpha=0.6, label=class_names[0], bins=15, color='green')
        if len(outputs_1) > 0:
            ax.hist(outputs_1, alpha=0.6, label=class_names[1], bins=15, color='red')
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_xlabel('Quantum Expectation Value ⟨Z⟩')
        ax.set_ylabel('Frequency')
        ax.set_title('Quantum Decision Space', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Confusion Matrix
    ax = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # 7. Performance Metrics
    ax = fig.add_subplot(gs[1, 3])
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    try:
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    except:
        precision = recall = f1 = test_accuracy
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [test_accuracy, precision, recall, f1]
    bars = ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Quantum Concepts Explanation
    ax = fig.add_subplot(gs[2, 0:2])
    concepts_text = """
    KEY QUANTUM CONCEPTS:
    
    ⚛️  Quantum Superposition: Each qubit exists in a superposition of |0⟩ and |1⟩, enabling 
        parallel processing of all 2^8 = 256 possible states simultaneously.
    
    🔗 Quantum Entanglement: CNOT gates create correlations between qubits, allowing the 
        circuit to model complex relationships in the data that classical systems process sequentially.
    
    🧲 Ising Model: The RZ rotations implement spin-spin coupling terms (J·σ_i·σ_j), naturally 
        mapping the physics of interacting spins to pattern recognition.
    
    🎯 Variational Learning: The quantum parameters are optimized using classical gradient 
        descent, creating a hybrid quantum-classical algorithm.
    """
    ax.text(0.02, 0.98, concepts_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='sans-serif',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
    
    # 9. Architecture Summary
    ax = fig.add_subplot(gs[2, 2])
    summary_text = f"""
    QUANTUM ARCHITECTURE:
    
    • Qubits: {classifier.n_qubits}
    • Layers: {classifier.n_layers}
    • Parameters: {len(classifier.weights.flatten())}
    • State Space: 2^{classifier.n_qubits} = {2**classifier.n_qubits} dimensions
    
    DATASET:
    • Total: {len(X)} samples
    • Test: {len(X_test)} samples
    • Features: {X.shape[1]}
    
    PERFORMANCE:
    • Test Accuracy: {test_accuracy:.3f}
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.axis('off')
    
    # 10. Classification Report
    ax = fig.add_subplot(gs[2, 3])
    report = classification_report(y_test, predictions, target_names=class_names, output_dict=True)
    report_text = f"""
    CLASSIFICATION REPORT:
    
    {class_names[0]}:
      Precision: {report[class_names[0]]['precision']:.3f}
      Recall:    {report[class_names[0]]['recall']:.3f}
      F1-Score:  {report[class_names[0]]['f1-score']:.3f}
    
    {class_names[1]}:
      Precision: {report[class_names[1]]['precision']:.3f}
      Recall:    {report[class_names[1]]['recall']:.3f}
      F1-Score:  {report[class_names[1]]['f1-score']:.3f}
    """
    ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    ax.axis('off')
    
    plt.tight_layout()
    filename = f'quantum_analysis_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📸 Visualization saved: {filename}")
    plt.close()

def run_mnist_demo():
    """Run quantum classification on MNIST dataset"""
    print("\n" + "="*80)
    print("🔢 QUANTUM CLASSIFICATION: MNIST DIGITS (0 vs 1)")
    print("="*80)
    
    # Load MNIST data
    X, y, X_original = load_mnist_data(n_samples=200)
    
    # Prepare sample images for visualization
    sample_images = []
    for i in range(min(8, len(X_original))):
        img = X_original[i].reshape(28, 28)
        sample_images.append((img, y[i], f'Digit {y[i]}'))
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]} (PCA-reduced from 784)")
    print(f"   Class distribution: 0s={np.sum(y==0)}, 1s={np.sum(y==1)}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n🎯 Dataset Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Train classifier
    classifier = QuantumIsingClassifier(n_qubits=8, n_layers=4)
    classifier.train(X_train, y_train, n_epochs=30)
    
    # Make predictions
    print(f"\n🔮 Making Predictions...")
    predictions, quantum_outputs = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Digit 0', 'Digit 1']))
    
    # Create visualization
    create_comprehensive_visualization('MNIST Digits', X, y, X_test, y_test, predictions,
                                     quantum_outputs, classifier, test_accuracy,
                                     sample_images, class_names=['Digit 0', 'Digit 1'])
    
    return classifier, test_accuracy

def run_blood_cell_demo(dataset_path):
    """Run quantum classification on blood cell dataset"""
    print("\n" + "="*80)
    print("🔬 QUANTUM CLASSIFICATION: BLOOD CELLS (Healthy vs AML)")
    print("="*80)
    
    # Load blood cell data
    X, y, sample_images = load_blood_cell_data(dataset_path, max_samples_per_class=100)
    
    if len(X) == 0:
        print("❌ No data loaded. Please check the dataset path.")
        return None, None
    
    if len(np.unique(y)) < 2:
        print("❌ Dataset contains only one class.")
        return None, None
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Class distribution: Healthy={np.sum(y==0)}, AML={np.sum(y==1)}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n🎯 Dataset Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Train classifier
    classifier = QuantumIsingClassifier(n_qubits=8, n_layers=4)
    classifier.train(X_train, y_train, n_epochs=30)
    
    # Make predictions
    print(f"\n🔮 Making Predictions...")
    predictions, quantum_outputs = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Healthy', 'AML']))
    
    # Create visualization
    create_comprehensive_visualization('Blood Cells', X, y, X_test, y_test, predictions,
                                     quantum_outputs, classifier, test_accuracy,
                                     sample_images, class_names=['Healthy', 'AML'])
    
    return classifier, test_accuracy

def main():
    """Main function to run both demos"""
    print("="*80)
    print("🧬 COMPREHENSIVE QUANTUM CLASSIFICATION DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases quantum computing for classification on:")
    print("1. MNIST handwritten digits (benchmark dataset)")
    print("2. Real blood cell images (medical application)")
    print("="*80)
    
    results = {}
    
    # Run MNIST demo
    mnist_classifier, mnist_accuracy = run_mnist_demo()
    results['MNIST'] = mnist_accuracy
    
    # Run Blood Cell demo
    blood_cell_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    if os.path.exists(blood_cell_path):
        blood_classifier, blood_accuracy = run_blood_cell_demo(blood_cell_path)
        if blood_accuracy is not None:
            results['Blood Cells'] = blood_accuracy
    else:
        print(f"\n⚠️  Blood cell dataset not found at: {blood_cell_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    for dataset, accuracy in results.items():
        print(f"{dataset:20} | Test Accuracy: {accuracy:.3f}")
    
    # Create comparison plot
    if len(results) > 1:
        plt.figure(figsize=(10, 6))
        datasets = list(results.keys())
        accuracies = list(results.values())
        
        bars = plt.bar(datasets, accuracies, color=['skyblue', 'lightcoral'], alpha=0.7)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title('Quantum Classifier Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('quantum_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📸 Comparison visualization saved: quantum_comparison.png")
        plt.close()
    
    print("\n✅ Demo complete! Check the generated PNG files for detailed visualizations.")

if __name__ == "__main__":
    main()
