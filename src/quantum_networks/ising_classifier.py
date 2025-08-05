import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import cv2
import glob

def load_images(path, label):
    data = []
    labels = []
    for img_file in glob.glob(f"{path}/*.tif"):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (16, 16)).flatten()  # Simplify for Ising
        data.append(img)
        labels.append(label)
    return np.array(data), np.array(labels)

def prepare_data():
    aml_path = 'data/aml_cytomorphology/raw/AML'
    control_path = 'data/aml_cytomorphology/raw/Control'

    aml_data, aml_labels = load_images(aml_path, 1)
    control_data, control_labels = load_images(control_path, 0)

    # Combine and encode labels
    data = np.concatenate((aml_data, control_data), axis=0)
    labels = np.concatenate((aml_labels, control_labels), axis=0)

    # Normalize data
    data = data / 255.0

    return train_test_split(data, labels, test_size=0.2, random_state=42)


def quantum_ising_classifier(X_train, y_train, X_test, y_test):
    num_qubits = 16  # For 16x16 image
    dev = qml.device('default.qubit', wires=num_qubits)

    def circuit(params, x=None):
        qml.templates.AngleEmbedding(x, wires=range(num_qubits))
        qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits))

    @qml.qnode(dev)
    def circuit_prob(params, x=None):
        circuit(params, x)
        return qml.probs(wires=range(num_qubits))

    # Define the cost function and optimizer
    def cost(params):
        probs = [circuit_prob(params, x=x) for x in X_train]
        pred = np.argmax(probs, axis=1)
        return np.mean(pred != y_train)

    # Initialize parameters
    init_params = 0.01 * np.random.randn(1, num_qubits, 3)

    # Optimize parameters
    opt = qml.GradientDescentOptimizer(stepsize=0.01)
    params = init_params
    for i in range(100):
        params = opt.step(cost, params)
        if (i + 1) % 10 == 0:
            print(f'Cost after {i+1} steps: {cost(params)}')

    # Evaluate on test data
    test_probs = [circuit_prob(params, x=x) for x in X_test]
    pred_labels = np.argmax(test_probs, axis=1)
    accuracy = np.mean(pred_labels == y_test)
    print(f'Test Accuracy: {accuracy}')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    quantum_ising_classifier(X_train, y_train, X_test, y_test)

