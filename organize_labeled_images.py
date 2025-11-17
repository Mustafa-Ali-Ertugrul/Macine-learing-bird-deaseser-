#!/usr/bin/env python3
"""
Organize labeled images from CSV into disease category folders
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configuration
CSV_FILE = 'poultry_labeled_12k.csv'
OUTPUT_DIR = 'organized_labeled_dataset'

# Disease mapping (standardize labels)
DISEASE_MAP = {
    'salmonella': 'salmonella',
    'coccidiosis': 'coccidiosis',
    'healthy': 'healthy',
    'ncd': 'newcastle',
    'pcrncd': 'newcastle',
    'pcrcocci': 'coccidiosis',
    'pcrsalmo': 'salmonella',
    'pcrhealthy': 'healthy',
    'unknown': 'unclassified',
    'ib': 'ib',
    'ibd': 'ibd',
    'fatty_liver': 'fatty_liver',
    'histomoniasis': 'histomoniasis',
    'marek': 'marek',
    'avian_influenza': 'avian_influenza'
}

def organize_images():
    """Organize images from CSV into disease folders"""
    
    print("=" * 70)
    print("ORGANIZE LABELED IMAGES INTO DISEASE CATEGORIES")
    print("=" * 70)
    
    # Load CSV
    print(f"\n[Step 1] Loading CSV: {CSV_FILE}")
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: CSV file not found: {CSV_FILE}")
        return
    
    df = pd.read_csv(CSV_FILE)
    print(f"   Total records: {len(df)}")
    
    # Create output directory structure
    print(f"\n[Step 2] Creating output directory: {OUTPUT_DIR}")
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Get unique diseases and create folders
    df['disease_clean'] = df['disease'].map(DISEASE_MAP)
    diseases = df['disease_clean'].dropna().unique()
    
    for disease in diseases:
        disease_dir = output_path / disease
        disease_dir.mkdir(exist_ok=True)
        print(f"   Created: {disease}")
    
    # Copy/move images
    print(f"\n[Step 3] Organizing images...")
    
    stats = {
        'copied': 0,
        'skipped_missing': 0,
        'skipped_no_label': 0,
        'errors': 0
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_path = row['image_path']
        disease = row['disease']
        
        # Map to clean disease name
        disease_clean = DISEASE_MAP.get(disease)
        
        if not disease_clean:
            stats['skipped_no_label'] += 1
            continue
        
        # Check if source image exists
        if not os.path.exists(image_path):
            stats['skipped_missing'] += 1
            continue
        
        try:
            # Create destination path
            filename = os.path.basename(image_path)
            dest_path = output_path / disease_clean / filename
            
            # Copy image (preserve original)
            shutil.copy2(image_path, dest_path)
            stats['copied'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:  # Only print first 5 errors
                print(f"\n   Error copying {image_path}: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ORGANIZATION COMPLETE")
    print("=" * 70)
    print(f"\n[Check] Images successfully copied: {stats['copied']}")
    print(f"   Missing source files: {stats['skipped_missing']}")
    print(f"   No disease label: {stats['skipped_no_label']}")
    print(f"   Errors: {stats['errors']}")
    
    # Count images per disease
    print(f"\n[Check] Images per disease category:")
    for disease in sorted(diseases):
        disease_dir = output_path / disease
        count = len(list(disease_dir.glob('*.jpg'))) + len(list(disease_dir.glob('*.png')))
        print(f"   {disease:<20} {count:>6} images")
    
    print(f"\n[Check] Total images: {stats['copied']}")
    print(f"\n[Check] Output directory: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("[Next] Update train_vit_b16.py to use 'data_dir': 'organized_labeled_dataset'")
    print("=" * 70)

if __name__ == '__main__':
    organize_images()
