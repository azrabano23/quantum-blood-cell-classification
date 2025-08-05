#!/usr/bin/env python3
"""
Quantum Ising Model for Blood Cell Classification
==================================================

This demonstration showcases the application of quantum computing concepts,
specifically quantum Ising models, to blood cell classification. The work
builds on quantum neural network research and equilibrium propagation techniques.

Author: A. Zrabano
Research Group: Dr. Liebovitch Lab
Focus: Quantum-Enhanced Medical Image Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class QuantumIsingBloodCellClassifier:
    """
    Quantum Ising Model for Blood Cell Classification
    
    This classifier uses quantum computing principles to distinguish between
    healthy blood cells and Acute Myeloid Leukemia (AML) cells using:
    
    1. Quantum superposition for parallel feature processing
    2. Ising spin interactions modeling cellular pattern correlations
    3. Variational quantum circuits for adaptive learning
    4. Quantum entanglement for complex pattern recognition
    """
    
    def __init__(self, n_qubits=16, n_layers=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.weights = None
        self.training_history = []
        
    def generate_synthetic_blood_cells(self, n_samples=200, image_size=4):
        """
        Generate synthetic blood cell data that mimics real characteristics:
        
        AML Cells (Malignant):
        - Irregular nuclear shape and size
        - Abnormal chromatin patterns
        - Larger nucleus-to-cytoplasm ratio
        - More cellular heterogeneity
        
        Healthy Cells:
        - Regular, round nuclear morphology
        - Uniform chromatin distribution
        - Normal nucleus-to-cytoplasm ratio
        - Consistent cellular appearance
        """
        print("Generating synthetic blood cell dataset...")
        print("Modeling key morphological differences:")
        print("• AML cells: Irregular nuclei, abnormal chromatin")
        print("• Healthy cells: Regular morphology, uniform patterns")
        
        # Generate AML cells (label 1) - irregular patterns
        aml_cells = []
        for i in range(n_samples//2):
            img = np.zeros((image_size, image_size))
            center = image_size // 2
            
            # Irregular nucleus characteristic of AML
            nucleus_irregularity = np.random.normal(0, 0.3)
            for x in range(image_size):
                for y in range(image_size):
                    dist = np.sqrt((x-center)**2 + (y-center)**2)
                    # Irregular boundary with noise
                    if dist < 1.5 + nucleus_irregularity + np.random.normal(0, 0.2):
                        # High nuclear intensity with heterogeneity
                        img[x,y] = 0.7 + np.random.normal(0, 0.15)
            
            # Add cytoplasmic abnormalities
            img += np.random.normal(0, 0.08, (image_size, image_size))
            aml_cells.append(img.flatten())
        
        # Generate healthy cells (label 0) - regular patterns
        healthy_cells = []
        for i in range(n_samples//2):
            img = np.zeros((image_size, image_size))
            center = image_size // 2
            
            # Regular circular nucleus
            for x in range(image_size):
                for y in range(image_size):
                    dist = np.sqrt((x-center)**2 + (y-center)**2)
                    if dist < 1.2:  # Consistent size
                        # Uniform nuclear intensity
                        img[x,y] = 0.5 + np.random.normal(0, 0.05)
            
            # Regular cytoplasm
            img += np.random.normal(0, 0.03, (image_size, image_size))
            healthy_cells.append(img.flatten())
        
        # Combine and normalize
        X = np.array(aml_cells + healthy_cells)
        y = np.array([1]*len(aml_cells) + [0]*len(healthy_cells))
        X = np.clip(X, 0, 1)
        
        return X, y
    
    def create_quantum_circuit(self):
        """
        Create quantum Ising model circuit with:
        
        1. Data Encoding Layer:
           - Maps pixel intensities to qubit rotation angles
           - Uses quantum superposition to encode all features simultaneously
        
        2. Ising Interaction Layers:
           - CNOT gates create entanglement between adjacent qubits
           - RZ rotations implement Ising coupling terms (J_ij * σ_z^i * σ_z^j)
           - Models spatial correlations in cellular patterns
        
        3. Local Field Layer:
           - RY rotations implement local magnetic fields (h_i * σ_y^i)
           - Allows individual qubit optimization
        
        4. Measurement:
           - Expectation value of Pauli-Z on first qubit
           - Maps quantum state to classical prediction
        """
        
        @qml.qnode(self.device)
        def quantum_circuit(weights, x):
            # Data Encoding: Map pixel intensities to quantum states
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(np.pi * x[i], wires=i)
            
            # Ising Interaction Layers
            for layer in range(self.n_layers):
                # Nearest neighbor Ising couplings
                for i in range(self.n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
                    qml.RZ(weights[layer, i], wires=i+1)
                    qml.CNOT(wires=[i, i+1])
                
                # Periodic boundary conditions (ring topology)
                qml.CNOT(wires=[self.n_qubits-1, 0])
                qml.RZ(weights[layer, self.n_qubits-1], wires=0)
                qml.CNOT(wires=[self.n_qubits-1, 0])
                
                # Local magnetic fields
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, self.n_qubits + i], wires=i)
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        return quantum_circuit
    
    def train(self, X_train, y_train, n_epochs=100, learning_rate=0.1):
        """Train the quantum Ising classifier using variational optimization"""
        
        print(f"\nTraining Quantum Ising Classifier...")
        print(f"Architecture: {self.n_qubits} qubits, {self.n_layers} layers")
        print(f"Parameters: {self.n_layers * (2 * self.n_qubits)} trainable weights")
        print("=" * 60)
        
        # Initialize weights
        self.weights = 0.01 * np.random.randn(self.n_layers, 2 * self.n_qubits)
        
        # Create quantum circuit
        circuit = self.create_quantum_circuit()
        
        # Cost function using quantum expectation values
        def cost_function(weights):
            total_cost = 0
            for x, y_true in zip(X_train, y_train):
                # Get quantum prediction
                quantum_output = circuit(weights, x)
                # Map to binary classification
                prediction = 1 if quantum_output > 0 else 0
                # Binary cross-entropy-like cost
                total_cost += abs(prediction - y_true)
            return total_cost / len(X_train)
        
        # Gradient-based optimizer
        optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)
        
        # Training loop
        self.training_history = []
        for epoch in range(n_epochs):
            self.weights = optimizer.step(cost_function, self.weights)
            cost = cost_function(self.weights)
            accuracy = 1 - cost
            self.training_history.append({'epoch': epoch, 'cost': cost, 'accuracy': accuracy})
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Cost = {cost:.4f}, Accuracy = {accuracy:.4f}")
        
        final_accuracy = 1 - cost_function(self.weights)
        print(f"\nTraining completed. Final accuracy: {final_accuracy:.4f}")
        return self.weights
    
    def predict(self, X):
        """Make predictions using the trained quantum classifier"""
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        circuit = self.create_quantum_circuit()
        predictions = []
        quantum_outputs = []
        
        for x in X:
            output = circuit(self.weights, x)
            quantum_outputs.append(output)
            prediction = 1 if output > 0 else 0
            predictions.append(prediction)
        
        return np.array(predictions), np.array(quantum_outputs)

def create_comprehensive_visualization(classifier, X, y, X_test, y_test, predictions, quantum_outputs):
    """Create comprehensive visualization of quantum classification results"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Sample Blood Cell Images
    ax1 = fig.add_subplot(gs[0, 0])
    image_size = int(np.sqrt(X.shape[1]))
    healthy_idx = np.where(y == 0)[0][0]
    ax1.imshow(X[healthy_idx].reshape(image_size, image_size), cmap='viridis', interpolation='nearest')
    ax1.set_title('Healthy Blood Cell\n(Synthetic)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    aml_idx = np.where(y == 1)[0][0]
    ax2.imshow(X[aml_idx].reshape(image_size, image_size), cmap='plasma', interpolation='nearest')
    ax2.set_title('AML Blood Cell\n(Synthetic)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 2. Quantum Circuit Visualization
    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.text(0.1, 0.8, 'Quantum Ising Model Architecture', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.65, '1. Data Encoding: Pixel intensities → Qubit rotations', fontsize=11)
    ax3.text(0.1, 0.55, '2. Ising Interactions: CNOT + RZ gates model spin couplings', fontsize=11)
    ax3.text(0.1, 0.45, '3. Local Fields: RY rotations for individual qubit control', fontsize=11)
    ax3.text(0.1, 0.35, '4. Measurement: Pauli-Z expectation → Classification', fontsize=11)
    ax3.text(0.1, 0.2, f'• {classifier.n_qubits} qubits, {classifier.n_layers} layers', fontsize=11, color='blue')
    ax3.text(0.1, 0.1, f'• {len(classifier.weights.flatten())} trainable parameters', fontsize=11, color='blue')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 3. Training History
    ax4 = fig.add_subplot(gs[1, 0])
    epochs = [h['epoch'] for h in classifier.training_history]
    costs = [h['cost'] for h in classifier.training_history]
    ax4.plot(epochs, costs, 'b-', linewidth=2, label='Training Cost')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Cost')
    ax4.set_title('Training Progress', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 4. Accuracy Progress
    ax5 = fig.add_subplot(gs[1, 1])
    accuracies = [h['accuracy'] for h in classifier.training_history]
    ax5.plot(epochs, accuracies, 'g-', linewidth=2, label='Training Accuracy')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Accuracy Progress', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 5. Confusion Matrix
    ax6 = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['Healthy', 'AML'], yticklabels=['Healthy', 'AML'])
    ax6.set_title('Confusion Matrix', fontweight='bold')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')
    
    # 6. Quantum Output Distribution
    ax7 = fig.add_subplot(gs[1, 3])
    healthy_outputs = quantum_outputs[y_test == 0]
    aml_outputs = quantum_outputs[y_test == 1]
    ax7.hist(healthy_outputs, alpha=0.6, label='Healthy', bins=10, color='green')
    ax7.hist(aml_outputs, alpha=0.6, label='AML', bins=10, color='red')
    ax7.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
    ax7.set_xlabel('Quantum Output')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Quantum Decision Distribution', fontweight='bold')
    ax7.legend()
    
    # 7. Performance Metrics
    ax8 = fig.add_subplot(gs[2, 0])
    test_accuracy = accuracy_score(y_test, predictions)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    from sklearn.metrics import precision_score, recall_score, f1_score
    values = [
        test_accuracy,
        precision_score(y_test, predictions, average='weighted'),
        recall_score(y_test, predictions, average='weighted'),
        f1_score(y_test, predictions, average='weighted')
    ]
    bars = ax8.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax8.set_ylabel('Score')
    ax8.set_title('Performance Metrics', fontweight='bold')
    ax8.set_ylim(0, 1)
    for bar, value in zip(bars, values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 8. Quantum Advantage Analysis
    ax9 = fig.add_subplot(gs[2, 1:])
    ax9.text(0.05, 0.9, 'Quantum Computing Advantages in Medical Image Analysis:', 
             fontsize=14, fontweight='bold')
    
    advantages = [
        "1. Superposition: Process all pixel features simultaneously in quantum parallel",
        "2. Entanglement: Model complex correlations between distant cellular regions", 
        "3. Ising Model: Natural mapping to spin systems captures cellular interactions",
        "4. Variational Learning: Adaptively optimize quantum parameters for classification",
        "5. Exponential State Space: Explore 2^n configurations with n qubits",
        "6. Quantum Interference: Constructive/destructive interference enhances patterns"
    ]
    
    for i, advantage in enumerate(advantages):
        ax9.text(0.05, 0.75 - i*0.1, advantage, fontsize=11, 
                color='darkblue' if i % 2 == 0 else 'darkgreen')
    
    ax9.text(0.05, 0.1, f'Test Accuracy: {test_accuracy:.3f} | Dataset: {len(X)} samples | '
             f'Qubits: {classifier.n_qubits} | Parameters: {len(classifier.weights.flatten())}',
             fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.suptitle('Quantum Ising Model for Blood Cell Classification\nDemonstrating Quantum-Enhanced Medical Image Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('quantum_blood_cell_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def main():
    """Main demonstration function"""
    
    print("="*80)
    print("QUANTUM ISING MODEL FOR BLOOD CELL CLASSIFICATION")
    print("="*80)
    print("Demonstrating quantum-enhanced medical image analysis")
    print("Research focus: Quantum neural networks for diagnostic applications")
    print("="*80)
    
    # Initialize quantum classifier
    classifier = QuantumIsingBloodCellClassifier(n_qubits=16, n_layers=4)
    
    # Generate synthetic blood cell dataset
    X, y = classifier.generate_synthetic_blood_cells(n_samples=120, image_size=4)
    
    print(f"\nDataset Overview:")
    print(f"• Total samples: {len(X)}")
    print(f"• Feature dimension: {X.shape[1]} (4x4 pixel images)")
    print(f"• Classes: Healthy (0) and AML (1)")
    print(f"• Healthy cells: {np.sum(y == 0)}")
    print(f"• AML cells: {np.sum(y == 1)}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"• Training samples: {len(X_train)}")
    print(f"• Test samples: {len(X_test)}")
    
    # Train quantum classifier
    classifier.train(X_train, y_train, n_epochs=100, learning_rate=0.1)
    
    # Make predictions
    predictions, quantum_outputs = classifier.predict(X_test)
    
    # Calculate performance metrics
    test_accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n" + "="*60)
    print("QUANTUM CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Quantum State Dimension: 2^{classifier.n_qubits} = {2**classifier.n_qubits:,}")
    print(f"Trainable Parameters: {len(classifier.weights.flatten())}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Healthy', 'AML']))
    
    # Create comprehensive visualization
    create_comprehensive_visualization(
        classifier, X, y, X_test, y_test, predictions, quantum_outputs
    )
    
    print(f"\n" + "="*60)
    print("QUANTUM COMPUTING IMPACT")
    print("="*60)
    print("This demonstration showcases:")
    print("• Quantum superposition for parallel feature processing")
    print("• Ising model mapping for cellular pattern recognition") 
    print("• Variational quantum algorithms for adaptive learning")
    print("• Entanglement-based correlation modeling")
    print("• Potential quantum advantage in medical diagnostics")
    
    print(f"\nVisualization saved: quantum_blood_cell_analysis.png")
    print(f"Ready for GitHub repository submission!")

if __name__ == "__main__":
    main()
