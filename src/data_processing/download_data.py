#!/usr/bin/env python3
"""
Data Download Script for Blood Cell Classification Datasets

This script downloads datasets from The Cancer Imaging Archive (TCIA) for
blood cell classification using quantum neural networks.

Usage:
    python download_data.py --dataset aml
    python download_data.py --dataset bone_marrow
    python download_data.py --dataset all
"""

import os
import sys
import requests
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm
import urllib.request

# Dataset URLs and information
DATASETS = {
    'aml': {
        'name': 'AML-Cytomorphology_LMU',
        'description': 'Acute Myeloid Leukemia Cytomorphology Dataset',
        'url': 'https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/',
        'download_url': None,  # TCIA requires registration - will provide instructions
        'size_gb': 11,
        'num_images': 18365,
        'classes': ['AML', 'Control'],
        'resolution': '100x magnification'
    },
    'bone_marrow': {
        'name': 'BONE-MARROW-CYTOMORPHOLOGY_MLL_HELMHOLTZ_FRAUNHOFER',
        'description': 'Bone Marrow Cytomorphology Dataset',
        'url': 'https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/',
        'download_url': None,  # TCIA requires registration
        'size_gb': 6.8,
        'num_images': 170000,
        'classes': ['Multiple cell types'],
        'resolution': '40x magnification'
    }
}

def print_dataset_info(dataset_key):
    """Print information about a dataset."""
    dataset = DATASETS[dataset_key]
    print(f"\n=== {dataset['name']} ===")
    print(f"Description: {dataset['description']}")
    print(f"Size: {dataset['size_gb']} GB")
    print(f"Number of images: {dataset['num_images']}")
    print(f"Classes: {', '.join(dataset['classes'])}")
    print(f"Resolution: {dataset['resolution']}")
    print(f"Info URL: {dataset['url']}")

def print_download_instructions():
    """Print instructions for downloading datasets from TCIA."""
    print("\n" + "="*80)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("""
The Cancer Imaging Archive (TCIA) datasets require registration and agreement
to their data use policies. To download the datasets:

1. Visit TCIA: https://www.cancerimagingarchive.net/
2. Create a free account if you don't have one
3. Download and install the NBIA Data Retriever tool
4. Navigate to the specific dataset collections:

   AML-Cytomorphology_LMU:
   https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/
   
   BONE-MARROW-CYTOMORPHOLOGY_MLL_HELMHOLTZ_FRAUNHOFER:
   https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/

5. Use the "Download" button to generate a .tcia file
6. Open the .tcia file with NBIA Data Retriever to download the datasets
7. Extract the downloaded data to the appropriate directories:
   - AML data -> data/aml_cytomorphology/
   - Bone marrow data -> data/bone_marrow_cytomorphology/

CITATION REQUIREMENTS:
Both datasets are available under CC BY 3.0 license and require proper citation.
See the dataset pages for specific citation information.

DATA USE AGREEMENT:
By downloading and using these datasets, you agree to:
- Use the data only for research purposes
- Properly cite the datasets in any publications
- Not redistribute the data without permission
- Follow all applicable ethics guidelines for medical data
""")
    print("="*80)

def create_data_structure():
    """Create the expected directory structure for datasets."""
    base_dir = Path("data")
    
    directories = [
        "aml_cytomorphology/raw",
        "aml_cytomorphology/processed",
        "bone_marrow_cytomorphology/raw", 
        "bone_marrow_cytomorphology/processed",
        "processed/train",
        "processed/val",
        "processed/test"
    ]
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")

def validate_data_structure():
    """Check if data directories exist and contain data."""
    base_dir = Path("data")
    
    aml_dir = base_dir / "aml_cytomorphology" / "raw"
    bone_marrow_dir = base_dir / "bone_marrow_cytomorphology" / "raw"
    
    print("\n=== DATA VALIDATION ===")
    
    if aml_dir.exists():
        aml_files = list(aml_dir.rglob("*"))
        print(f"AML dataset directory exists with {len(aml_files)} files/folders")
    else:
        print("AML dataset directory not found")
    
    if bone_marrow_dir.exists():
        bm_files = list(bone_marrow_dir.rglob("*"))
        print(f"Bone marrow dataset directory exists with {len(bm_files)} files/folders")
    else:
        print("Bone marrow dataset directory not found")

def main():
    parser = argparse.ArgumentParser(description="Download blood cell datasets from TCIA")
    parser.add_argument('--dataset', choices=['aml', 'bone_marrow', 'all'], 
                       default='all', help='Dataset to download')
    parser.add_argument('--info-only', action='store_true', 
                       help='Only show dataset information')
    parser.add_argument('--setup', action='store_true',
                       help='Create directory structure only')
    
    args = parser.parse_args()
    
    if args.setup:
        create_data_structure()
        return
    
    if args.dataset == 'all':
        datasets_to_process = ['aml', 'bone_marrow']
    else:
        datasets_to_process = [args.dataset]
    
    # Show dataset information
    for dataset_key in datasets_to_process:
        print_dataset_info(dataset_key)
    
    if args.info_only:
        return
    
    # Create directory structure
    create_data_structure()
    
    # Show download instructions
    print_download_instructions()
    
    # Validate existing data
    validate_data_structure()

if __name__ == "__main__":
    main()
