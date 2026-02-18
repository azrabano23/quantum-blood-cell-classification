#!/usr/bin/env python3
"""
Shared Feature Extractor for Blood Cell Classification
======================================================

Comprehensive feature extraction used by all classifiers.
Extracts 32 features covering statistical, texture, morphological,
edge, and frequency domain characteristics.

Author: A. Zrabano
"""

import numpy as np
from skimage.io import imread_collection
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel, laplace
from skimage.measure import regionprops, label
from scipy import ndimage
from scipy.fftpack import fft2, fftshift
import warnings
warnings.filterwarnings('ignore')


def extract_features(img_path, n_features=32):
    """
    Extract comprehensive features from a blood cell image.
    
    Features (32 total):
    - Statistical (8): mean, std, median, Q25, Q75, range, skewness, kurtosis
    - GLCM Texture (10): contrast, dissimilarity, homogeneity, energy, correlation, ASM (mean+std)
    - Morphological (6): area ratio, eccentricity, solidity, extent, perimeter ratio, circularity
    - Edge (4): sobel mean/std, laplacian mean/std
    - Frequency (4): low/high frequency energy, spectral centroid, spectral spread
    
    Args:
        img_path: Path to image file
        n_features: Number of features to return (default 32)
    
    Returns:
        numpy array of features
    """
    try:
        img = imread_collection([img_path])[0]
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img
        
        # Resize to standard size
        img_resized = resize(img_gray, (64, 64), anti_aliasing=True)
        img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
        
        features = []
        
        # ============ 1. Statistical Features (8) ============
        features.append(np.mean(img_norm))
        features.append(np.std(img_norm))
        features.append(np.median(img_norm))
        features.append(np.percentile(img_norm, 25))
        features.append(np.percentile(img_norm, 75))
        features.append(np.max(img_norm) - np.min(img_norm))
        # Skewness
        mean_val = np.mean(img_norm)
        std_val = np.std(img_norm) + 1e-8
        features.append(np.mean(((img_norm - mean_val) / std_val) ** 3))
        # Kurtosis
        features.append(np.mean(((img_norm - mean_val) / std_val) ** 4) - 3)
        
        # ============ 2. GLCM Texture Features (10) ============
        img_uint8 = (img_norm * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                          levels=256, symmetric=True, normed=True)
        
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            vals = graycoprops(glcm, prop)
            features.append(np.mean(vals))
        features.append(np.mean(graycoprops(glcm, 'ASM')))
        # Add std of key properties for more discrimination
        features.append(np.std(graycoprops(glcm, 'contrast')))
        features.append(np.std(graycoprops(glcm, 'homogeneity')))
        features.append(np.std(graycoprops(glcm, 'energy')))
        features.append(np.std(graycoprops(glcm, 'correlation')))
        
        # ============ 3. Morphological Features (6) ============
        # Threshold to get cell region
        thresh = img_norm > np.mean(img_norm)
        labeled = label(thresh)
        
        if labeled.max() > 0:
            # Get largest region
            props = sorted(regionprops(labeled), key=lambda x: x.area, reverse=True)[0]
            features.append(props.area / (64 * 64))  # Normalized area
            features.append(props.eccentricity)  # Shape elongation
            features.append(props.solidity)  # Convexity
            features.append(props.extent)  # Bounding box fill
            features.append(props.perimeter / (2 * np.pi * np.sqrt(props.area / np.pi) + 1e-8))  # Perimeter ratio
            # Circularity
            features.append(4 * np.pi * props.area / (props.perimeter ** 2 + 1e-8))
        else:
            features.extend([0.5, 0.5, 0.5, 0.5, 1.0, 1.0])
        
        # ============ 4. Edge Features (4) ============
        edges_sobel = sobel(img_norm)
        edges_lap = np.abs(laplace(img_norm))
        
        features.append(np.mean(edges_sobel))
        features.append(np.std(edges_sobel))
        features.append(np.mean(edges_lap))
        features.append(np.std(edges_lap))
        
        # ============ 5. Frequency Domain Features (4) ============
        f_transform = fft2(img_norm)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Create frequency masks
        rows, cols = img_norm.shape
        crow, ccol = rows // 2, cols // 2
        
        # Low frequency (center)
        low_mask = np.zeros((rows, cols))
        low_mask[crow-8:crow+8, ccol-8:ccol+8] = 1
        low_energy = np.sum(magnitude * low_mask) / (np.sum(magnitude) + 1e-8)
        
        # High frequency (edges)
        high_mask = 1 - low_mask
        high_energy = np.sum(magnitude * high_mask) / (np.sum(magnitude) + 1e-8)
        
        features.append(low_energy)
        features.append(high_energy)
        
        # Spectral centroid and spread
        freq_x = np.arange(cols) - ccol
        freq_y = np.arange(rows) - crow
        fx, fy = np.meshgrid(freq_x, freq_y)
        freq_dist = np.sqrt(fx**2 + fy**2)
        
        total_mag = np.sum(magnitude) + 1e-8
        spectral_centroid = np.sum(freq_dist * magnitude) / total_mag
        spectral_spread = np.sqrt(np.sum(((freq_dist - spectral_centroid) ** 2) * magnitude) / total_mag)
        
        features.append(spectral_centroid / 32)  # Normalize
        features.append(spectral_spread / 32)
        
        return np.array(features[:n_features])
        
    except Exception as e:
        # Return zeros on error (will be handled by scaler)
        return np.zeros(n_features)


def load_dataset(dataset_folder, max_samples_per_class=150, n_features=32):
    """
    Load blood cell dataset with extracted features.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (0=healthy, 1=AML)
    """
    import os
    
    print(f"Loading data from: {dataset_folder}")
    
    healthy_cell_types = ['LYT', 'MON', 'NGS', 'NGB']
    aml_cell_types = ['MYB', 'MOB', 'MMZ', 'KSC', 'BAS', 'EBO', 'EOS', 'LYA', 'MYO', 'PMO']
    
    X, y = [], []
    class_counts = {'healthy': 0, 'aml': 0}
    
    for dirpath, _, filenames in os.walk(dataset_folder):
        path_parts = dirpath.split(os.sep)
        cell_type = None
        
        for part in path_parts:
            if part in healthy_cell_types or part in aml_cell_types:
                cell_type = part
                break
        
        if cell_type is None:
            continue
        
        for file in filenames:
            if file.endswith(('.jpg', '.png', '.tiff', '.tif')):
                if cell_type in healthy_cell_types:
                    if class_counts['healthy'] >= max_samples_per_class:
                        continue
                    label_val = 0
                    class_counts['healthy'] += 1
                elif cell_type in aml_cell_types:
                    if class_counts['aml'] >= max_samples_per_class:
                        continue
                    label_val = 1
                    class_counts['aml'] += 1
                else:
                    continue
                
                img_path = os.path.join(dirpath, file)
                features = extract_features(img_path, n_features)
                X.append(features)
                y.append(label_val)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples: Healthy={class_counts['healthy']}, AML={class_counts['aml']}")
    print(f"Features per sample: {n_features}")
    
    return X, y


if __name__ == "__main__":
    # Test feature extraction
    import sys
    
    dataset_path = "/Users/azrabano/Downloads/PKG - AML-Cytomorphology_LMU"
    X, y = load_dataset(dataset_path, max_samples_per_class=10)
    print(f"Feature shape: {X.shape}")
    print(f"Sample features: {X[0][:5]}...")
