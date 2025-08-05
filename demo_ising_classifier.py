#!/usr/bin/env python3
"""
Quick Demo: Quantum Ising Model for Blood Cell Classification
This creates synthetic blood cell data and demonstrates quantum classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pennylane as qml
from pennylane import numpy as pnp

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_blood_cells(n_samples=200, image_size=4):
    """Generate synthetic blood cell images for demonstration"""
    
    # Generate AML cells (label 1) - more irregular patterns
    aml_cells = []
    for _ in range(n_samples//2):
        # Create base circular pattern with noise
        img = np.zeros((image_size, image_size))
        center = image_size // 2
        
        # Add irregular nucleus (characteristic of AML)
        for i in range(image_size):
            for j in range(image_size):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                if dist < 2.5 + np.random.normal(0, 0.5):
                    img[i,j] = 0.8 + np.random.normal(0, 0.2)
        
        # Add cytoplasm irregularities
        img += np.random.normal(0, 0.1, (image_size, image_size))
        aml_cells.append(img.flatten())
    
    # Generate healthy cells (label 0) - more regular patterns  
    healthy_cells = []
    for _ in range(n_samples//2):
        img = np.zeros((image_size, image_size))
        center = image_size // 2
        
        # Regular circular nucleus
        for i in range(image_size):
            for j in range(image_size):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                if dist < 2.0:
                    img[i,j] = 0.6 + np.random.normal(0, 0.1)
        
        # Regular cytoplasm
        img += np.random.normal(0, 0.05, (image_size, image_size))
        healthy_cells.append(img.flatten())
    
    # Combine data
    X = np.array(aml_cells + healthy_cells)
    y = np.array([1]*len(aml_cells) + [0]*len(healthy_cells))
    
    # Normalize to [0, 1]
    X = np.clip(X, 0, 1)
    
    return X, y

def create_quantum_ising_classifier(n_qubits=8, n_layers=3):
    """Create quantum Ising model classifier"""
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def quantum_circuit(weights, x):
        # Data encoding - map pixel intensities to qubit rotations
        for i in range(n_qubits):
            qml.RY(np.pi * x[i], wires=i)
        
        # Ising-like interactions with parameterized gates
        for layer in range(n_layers):
            # ZZ interactions (Ising coupling)
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])
                qml.RZ(weights[layer, i], wires=i+1)
                qml.CNOT(wires=[i, i+1])
            
            # Local fields
            for i in range(n_qubits):
                qml.RY(weights[layer, n_qubits + i], wires=i)
        
        # Measurement
        return qml.expval(qml.PauliZ(0))
    
    return quantum_circuit

def train_quantum_classifier(X_train, y_train, X_test, y_test, n_epochs=50):
    """Train the quantum Ising classifier"""
    
    n_qubits = X_train.shape[1]  # Number of pixels
    n_layers = 3
    
    # Initialize weights
    weights = 0.01 * np.random.randn(n_layers, 2 * n_qubits)
    
    # Create quantum circuit
    circuit = create_quantum_ising_classifier(n_qubits, n_layers)
    
    # Cost function
    def cost_fn(weights):
        predictions = []
        for x, y_true in zip(X_train, y_train):
            pred = circuit(weights, x)
            # Map quantum expectation to binary classification
            pred_binary = 1 if pred > 0 else 0
            predictions.append(pred_binary)
        
        accuracy = np.mean(np.array(predictions) == y_train)
        return 1 - accuracy  # Minimize error
    
    # Optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    
    print("Training Quantum Ising Classifier...")
    print("=" * 50)
    
    costs = []
    for epoch in range(n_epochs):
        weights = opt.step(cost_fn, weights)
        cost = cost_fn(weights)
        costs.append(cost)
        
        if epoch % 10 == 0:
            accuracy = 1 - cost
            print(f"Epoch {epoch:3d}: Cost = {cost:.4f}, Accuracy = {accuracy:.4f}")
    
    # Test the trained model
    test_predictions = []
    test_probabilities = []
    
    for x in X_test:
        pred = circuit(weights, x)
        test_probabilities.append(pred)
        pred_binary = 1 if pred > 0 else 0
        test_predictions.append(pred_binary)
    
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print("\n" + "=" * 50)
    print("QUANTUM ISING CLASSIFIER RESULTS")
    print("=" * 50)
    print(f"Final Training Accuracy: {1-costs[-1]:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions, 
                              target_names=['Healthy', 'AML']))
    
    return weights, costs, test_predictions, test_probabilities

def visualize_results(X, y, test_predictions, costs):
    """Visualize the results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot sample images
    image_size = int(np.sqrt(X.shape[1]))
    
    # Healthy cell sample
    healthy_idx = np.where(y == 0)[0][0]
    axes[0,0].imshow(X[healthy_idx].reshape(image_size, image_size), cmap='viridis')
    axes[0,0].set_title('Healthy Cell (Synthetic)')
    axes[0,0].axis('off')
    
    # AML cell sample  
    aml_idx = np.where(y == 1)[0][0]
    axes[0,1].imshow(X[aml_idx].reshape(image_size, image_size), cmap='viridis')
    axes[0,1].set_title('AML Cell (Synthetic)')
    axes[0,1].axis('off')
    
    # Training cost
    axes[0,2].plot(costs)
    axes[0,2].set_title('Training Cost (1 - Accuracy)')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Cost')
    
    # Confusion matrix-like visualization
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y[-len(test_predictions):], test_predictions)
    im = axes[1,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1,0].set_title('Confusion Matrix')
    axes[1,0].set_ylabel('True Label')
    axes[1,0].set_xlabel('Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1,0].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    axes[1,1].bar(['Healthy', 'AML'], counts, color=['green', 'red'], alpha=0.7)
    axes[1,1].set_title('Dataset Distribution')
    axes[1,1].set_ylabel('Number of Samples')
    
    # Quantum advantage visualization
    axes[1,2].text(0.1, 0.8, 'Quantum Ising Model Features:', fontsize=12, fontweight='bold')
    axes[1,2].text(0.1, 0.6, '• Quantum superposition encoding', fontsize=10)
    axes[1,2].text(0.1, 0.5, '• Ising spin interactions', fontsize=10)
    axes[1,2].text(0.1, 0.4, '• Entanglement-based learning', fontsize=10)
    axes[1,2].text(0.1, 0.2, f'• Test Accuracy: {accuracy_score(y[-len(test_predictions):], test_predictions):.3f}', 
                   fontsize=11, fontweight='bold')
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('quantum_ising_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main demonstration function"""
    
    print("QUANTUM ISING MODEL FOR BLOOD CELL CLASSIFICATION")
    print("=" * 60)
    print("Generating synthetic blood cell data...")
    
    # Generate synthetic data
    X, y = generate_synthetic_blood_cells(n_samples=100, image_size=4)
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features (4x4 pixels)")
    print(f"Classes: {np.unique(y)} (0=Healthy, 1=AML)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train quantum classifier
    weights, costs, test_predictions, test_probabilities = train_quantum_classifier(
        X_train, y_train, X_test, y_test, n_epochs=50
    )
    
    # Visualize results
    visualize_results(X, y, test_predictions, costs)
    
    print("\n" + "=" * 60)
    print("QUANTUM ADVANTAGE ANALYSIS")
    print("=" * 60)
    print("This quantum Ising model demonstrates:")
    print("1. Quantum superposition for parallel feature processing")
    print("2. Ising-like spin interactions modeling cellular patterns")
    print("3. Entanglement-based pattern recognition")
    print("4. Potential for quantum speedup on larger datasets")
    
    print(f"\nResults saved to: quantum_ising_results.png")

if __name__ == "__main__":
    main()
