#!/usr/bin/env python3
"""
Quantum Equilibrium Propagation Implementation
==============================================

This module implements Equilibrium Propagation (EP) for quantum systems,
a training framework that leverages the system's physics to perform 
gradient-based learning in quantum neural networks.

Equilibrium Propagation is based on the principle that the system
naturally evolves to an equilibrium state, and gradients can be
computed by perturbing this equilibrium.

Author: A. Zrabano
Research Group: Dr. Liebovitch Lab
Focus: Quantum-Enhanced Medical Image Analysis
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import time

class QuantumEquilibriumPropagation:
    """
    Quantum Equilibrium Propagation Classifier
    
    This implementation uses the principles of Equilibrium Propagation
    to train quantum circuits by:
    
    1. Finding equilibrium states of the quantum system
    2. Computing gradients through equilibrium perturbations
    3. Using energy-based learning for classification
    """
    
    def __init__(self, n_qubits=8, n_layers=3, beta=1.0):
        """
        Initialize Quantum Equilibrium Propagation
        
        Args:
            n_qubits (int): Number of qubits in the quantum circuit
            n_layers (int): Number of variational layers
            beta (float): Inverse temperature parameter for equilibrium
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.beta = beta
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.weights = None
        self.training_history = []
        
    def create_energy_circuit(self):
        """
        Create quantum circuit for energy-based learning
        
        The energy function E(θ, x) is defined as the expectation value
        of a Hamiltonian that depends on the input data and parameters.
        """
        
        @qml.qnode(self.device)
        def energy_circuit(weights, x):
            # Data encoding layer
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(np.pi * x[i], wires=i)
            
            # Variational layers for energy landscape
            for layer in range(self.n_layers):
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                    qml.RZ(weights[layer, self.n_qubits + i], wires=i)
            
            # Energy measurement - expectation value of Pauli-Z on all qubits
            return qml.expval(qml.Hamiltonian(
                [1.0] * self.n_qubits,
                [qml.PauliZ(i) for i in range(self.n_qubits)]
            ))
        
        return energy_circuit
    
    def create_classification_circuit(self):
        """
        Create quantum circuit for classification output
        
        This circuit uses the energy-based approach to make predictions
        by comparing energy values for different classes.
        """
        
        @qml.qnode(self.device)
        def classification_circuit(weights, x):
            # Data encoding
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(np.pi * x[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                    qml.RZ(weights[layer, self.n_qubits + i], wires=i)
            
            # Classification measurement
            return qml.expval(qml.PauliZ(0))
        
        return classification_circuit
    
    def equilibrium_state(self, weights, x, target_class=None):
        """
        Find the equilibrium state of the quantum system
        
        In Equilibrium Propagation, we find the state that minimizes
        the energy function E(θ, x) + β * L(y, ŷ) where L is the loss.
        """
        energy_circuit = self.create_energy_circuit()
        
        # Compute base energy
        base_energy = energy_circuit(weights, x)
        
        # If target class is provided, add classification loss
        if target_class is not None:
            classification_circuit = self.create_classification_circuit()
            prediction = classification_circuit(weights, x)
            
            # Binary classification loss
            if target_class == 1:
                loss = -prediction  # Want positive prediction for class 1
            else:
                loss = prediction   # Want negative prediction for class 0
            
            total_energy = base_energy + self.beta * loss
        else:
            total_energy = base_energy
        
        return total_energy
    
    def compute_equilibrium_gradient(self, weights, x, y_true):
        """
        Compute gradients using Equilibrium Propagation
        
        The key insight of EP is that gradients can be computed by
        comparing the equilibrium states with and without the loss term.
        """
        
        # Forward pass: equilibrium with loss
        energy_with_loss = self.equilibrium_state(weights, x, y_true)
        
        # Free phase: equilibrium without loss
        energy_free = self.equilibrium_state(weights, x, None)
        
        # Compute gradient using parameter shift rule
        gradient = np.zeros_like(weights)
        
        for layer in range(self.n_layers):
            for param_idx in range(weights.shape[1]):
                # Parameter shift rule
                weights_plus = weights.copy()
                weights_minus = weights.copy()
                
                weights_plus[layer, param_idx] += np.pi/2
                weights_minus[layer, param_idx] -= np.pi/2
                
                # Compute energies with shifted parameters
                energy_plus = self.equilibrium_state(weights_plus, x, y_true)
                energy_minus = self.equilibrium_state(weights_minus, x, y_true)
                
                # Gradient approximation
                gradient[layer, param_idx] = 0.5 * (energy_plus - energy_minus)
        
        return gradient
    
    def train(self, X_train, y_train, n_epochs=50, learning_rate=0.1):
        """
        Train the quantum circuit using Equilibrium Propagation
        
        Args:
            X_train: Training data
            y_train: Training labels
            n_epochs: Number of training epochs
            learning_rate: Learning rate for parameter updates
        """
        
        print(f"\n⚛️  Training Quantum Equilibrium Propagation")
        print(f"   Architecture: {self.n_qubits} qubits, {self.n_layers} layers")
        print(f"   Beta (inverse temperature): {self.beta}")
        print(f"   Training samples: {len(X_train)}")
        
        # Initialize parameters
        self.weights = 0.01 * np.random.randn(self.n_layers, 2 * self.n_qubits)
        
        self.training_history = []
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Compute gradients for all training samples
            total_gradient = np.zeros_like(self.weights)
            
            for x, y_true in zip(X_train, y_train):
                gradient = self.compute_equilibrium_gradient(self.weights, x[:self.n_qubits], y_true)
                total_gradient += gradient
            
            # Average gradient
            total_gradient /= len(X_train)
            
            # Update parameters
            self.weights -= learning_rate * total_gradient
            
            # Compute training accuracy
            predictions = self.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            
            epoch_time = time.time() - epoch_start
            
            self.training_history.append({
                'epoch': epoch,
                'accuracy': accuracy,
                'time': epoch_time
            })
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:2d}: Accuracy = {accuracy:.3f}, Time = {epoch_time:.2f}s")
        
        final_accuracy = accuracy_score(y_train, self.predict(X_train))
        print(f"   Training complete: Final accuracy = {final_accuracy:.3f}")
        
        return self.weights
    
    def predict(self, X):
        """
        Make predictions using the trained quantum circuit
        
        Args:
            X: Input data
            
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        classification_circuit = self.create_classification_circuit()
        predictions = []
        
        for x in X:
            output = classification_circuit(self.weights, x[:self.n_qubits])
            pred = 1 if output > 0 else 0
            predictions.append(pred)
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'classification_report': classification_report(y_test, predictions)
        }
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.training_history:
            print("No training history available")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        accuracies = [h['accuracy'] for h in self.training_history]
        times = [h['time'] for h in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(epochs, accuracies, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title('Equilibrium Propagation Training Progress')
        ax1.grid(True, alpha=0.3)
        
        # Time plot
        ax2.plot(epochs, times, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time per Epoch')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def compare_with_vqc(self, vqc_results):
        """
        Compare Equilibrium Propagation results with VQC results
        
        Args:
            vqc_results: Results from Variational Quantum Classifier
            
        Returns:
            dict: Comparison metrics
        """
        comparison = {
            'method': 'Equilibrium Propagation vs VQC',
            'ep_accuracy': self.training_history[-1]['accuracy'] if self.training_history else 0,
            'vqc_accuracy': vqc_results.get('accuracy', 0),
            'ep_training_time': sum([h['time'] for h in self.training_history]),
            'vqc_training_time': vqc_results.get('training_time', 0),
            'ep_convergence': len(self.training_history),
            'vqc_convergence': vqc_results.get('epochs', 0)
        }
        
        return comparison
