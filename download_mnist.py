#!/usr/bin/env python3
"""
MNIST Dataset Downloader and Preparer
=====================================

This script downloads the MNIST dataset from Kaggle and prepares it
for quantum machine learning experiments.

Author: A. Zrabano
Research Group: Dr. Liebovitch Lab
Focus: Quantum-Enhanced Medical Image Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile
import requests
from io import BytesIO

def download_mnist_from_kaggle():
    """
    Download MNIST dataset from Kaggle
    Note: This requires Kaggle API credentials
    """
    print("📥 Downloading MNIST dataset from Kaggle...")
    
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files('hojjatk/mnist-dataset', path='./data/mnist', unzip=True)
        
        print("✅ MNIST dataset downloaded successfully!")
        return True
        
    except ImportError:
        print("⚠️  Kaggle API not installed. Installing...")
        os.system("pip install kaggle")
        return download_mnist_from_kaggle()
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return False

def create_synthetic_mnist():
    """
    Create a synthetic MNIST-like dataset for testing
    """
    print("🎨 Creating synthetic MNIST-like dataset...")
    
    # Create synthetic 28x28 images
    n_samples = 2000
    image_size = 28
    
    # Generate random images with some structure
    X = np.random.rand(n_samples, image_size * image_size)
    
    # Add some structure to make it more realistic
    for i in range(n_samples):
        # Add some patterns
        img = X[i].reshape(image_size, image_size)
        
        # Add some circular patterns
        center_x, center_y = np.random.randint(5, 23, 2)
        y, x = np.ogrid[:image_size, :image_size]
        mask = (x - center_x)**2 + (y - center_y)**2 < np.random.randint(3, 8)**2
        img[mask] = np.random.rand()
        
        # Add some linear patterns
        if np.random.rand() > 0.5:
            start_x, start_y = np.random.randint(0, 20, 2)
            end_x, end_y = np.random.randint(8, 28, 2)
            for t in np.linspace(0, 1, 20):
                x_coord = int(start_x + t * (end_x - start_x))
                y_coord = int(start_y + t * (end_y - start_y))
                if 0 <= x_coord < image_size and 0 <= y_coord < image_size:
                    img[y_coord, x_coord] = np.random.rand()
        
        X[i] = img.flatten()
    
    # Create labels (binary classification: even vs odd digits)
    y = np.random.randint(0, 10, n_samples)
    y_binary = (y % 2)  # 0 for even, 1 for odd
    
    print(f"✅ Created {n_samples} synthetic MNIST samples")
    print(f"   Image shape: {image_size}x{image_size}")
    print(f"   Class distribution: {np.bincount(y_binary)}")
    
    return X, y_binary

def load_mnist_data():
    """
    Load MNIST data (either from Kaggle or synthetic)
    """
    # Try to load from Kaggle first
    if os.path.exists('./data/mnist/train.csv'):
        print("📊 Loading MNIST from downloaded files...")
        
        # Load training data
        train_df = pd.read_csv('./data/mnist/train.csv')
        test_df = pd.read_csv('./data/mnist/test.csv')
        
        # Extract features and labels
        X_train = train_df.iloc[:, 1:].values  # All columns except first (label)
        y_train = train_df.iloc[:, 0].values   # First column (label)
        
        # For test data, we don't have labels, so we'll use part of training data
        X_test = test_df.values
        
        # Convert to binary classification (even vs odd digits)
        y_train_binary = (y_train % 2)
        
        # Use part of training data as test data
        X_train, X_test, y_train_binary, y_test_binary = train_test_split(
            X_train, y_train_binary, test_size=0.2, random_state=42, stratify=y_train_binary
        )
        
        print(f"✅ Loaded MNIST dataset:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        print(f"   Class distribution: {np.bincount(y_train_binary)}")
        
        return X_train, X_test, y_train_binary, y_test_binary
    
    else:
        print("📊 Using synthetic MNIST dataset...")
        X, y = create_synthetic_mnist()
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test

def prepare_mnist_for_quantum(X_train, X_test, y_train, y_test, target_size=8):
    """
    Prepare MNIST data for quantum processing
    
    Args:
        X_train, X_test: Image data
        y_train, y_test: Labels
        target_size: Target image size for quantum processing
    """
    print(f"🔧 Preparing MNIST data for quantum processing...")
    print(f"   Resizing from 28x28 to {target_size}x{target_size}")
    
    # Resize images
    def resize_images(X, target_size):
        X_resized = []
        for img in X:
            img_2d = img.reshape(28, 28)
            # Simple downsampling
            step = 28 // target_size
            img_resized = img_2d[::step, ::step][:target_size, :target_size]
            X_resized.append(img_resized.flatten())
        return np.array(X_resized)
    
    X_train_resized = resize_images(X_train, target_size)
    X_test_resized = resize_images(X_test, target_size)
    
    # Normalize to [0, 1]
    X_train_resized = X_train_resized / 255.0
    X_test_resized = X_test_resized / 255.0
    
    print(f"✅ Data prepared:")
    print(f"   Training shape: {X_train_resized.shape}")
    print(f"   Test shape: {X_test_resized.shape}")
    print(f"   Value range: [{X_train_resized.min():.3f}, {X_train_resized.max():.3f}]")
    
    return X_train_resized, X_test_resized, y_train, y_test

def create_data_splits(X_train, X_test, y_train, y_test):
    """
    Create the required data splits: 200/100 and 1000/100
    """
    print("📊 Creating data splits for experiments...")
    
    # Split 1: 200 train, 100 test
    X_train_200 = X_train[:200]
    y_train_200 = y_train[:200]
    X_test_100 = X_test[:100]
    y_test_100 = y_test[:100]
    
    # Split 2: 1000 train, 100 test
    X_train_1000 = X_train[:1000]
    y_train_1000 = y_train[:1000]
    X_test_100_2 = X_test[:100]
    y_test_100_2 = y_test[:100]
    
    splits = {
        'split_200_100': {
            'X_train': X_train_200,
            'y_train': y_train_200,
            'X_test': X_test_100,
            'y_test': y_test_100
        },
        'split_1000_100': {
            'X_train': X_train_1000,
            'y_train': y_train_1000,
            'X_test': X_test_100_2,
            'y_test': y_test_100_2
        }
    }
    
    print("✅ Data splits created:")
    print(f"   Split 1: {len(X_train_200)} train, {len(X_test_100)} test")
    print(f"   Split 2: {len(X_train_1000)} train, {len(X_test_100_2)} test")
    
    return splits

def save_mnist_data(splits, save_dir='./data/mnist_processed'):
    """
    Save processed MNIST data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        np.savez(f'{save_dir}/{split_name}.npz', **split_data)
        print(f"💾 Saved {split_name} to {save_dir}/{split_name}.npz")
    
    print(f"✅ All data splits saved to {save_dir}/")

def visualize_mnist_samples(X, y, n_samples=8, title="MNIST Samples"):
    """
    Visualize MNIST samples
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    for i in range(n_samples):
        row = i // 4
        col = i % 4
        
        img = X[i].reshape(8, 8)  # Assuming 8x8 images
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Label: {y[i]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to download and prepare MNIST data
    """
    print("🔢 MNIST Dataset Preparation for Quantum Experiments")
    print("=" * 60)
    
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Load MNIST data
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Prepare for quantum processing
    X_train_q, X_test_q, y_train_q, y_test_q = prepare_mnist_for_quantum(
        X_train, X_test, y_train, y_test, target_size=8
    )
    
    # Create data splits
    splits = create_data_splits(X_train_q, X_test_q, y_train_q, y_test_q)
    
    # Save data
    save_mnist_data(splits)
    
    # Visualize samples
    visualize_mnist_samples(
        splits['split_200_100']['X_train'], 
        splits['split_200_100']['y_train'],
        title="MNIST Training Samples (200/100 Split)"
    )
    
    print("\n🎯 MNIST dataset preparation complete!")
    print("   Ready for quantum experiments in Google Colab")

if __name__ == "__main__":
    main()
