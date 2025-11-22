import pandas as pd
import os
from pathlib import Path

print("=" * 60)
print("POULTRY DISEASE DATASET ANALYSIS")
print("=" * 60)

# Analyze CSV metadata
csv_file = "poultry_labeled_12k.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    
    print(f"\n📊 CSV Metadata ({csv_file}):")
    print(f"   Total records: {len(df):,}")
    
    labeled = df[df['disease'] != 'unknown']
    unlabeled = df[df['disease'] == 'unknown']
    
    print(f"   Labeled: {len(labeled):,}")
    print(f"   Unlabeled: {len(unlabeled):,}")
    
    print("\n📈 Disease Distribution:")
    disease_counts = df['disease'].value_counts()
    for disease, count in disease_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {disease:15} {count:5,} images ({pct:5.1f}%)")

# Analyze image files on disk
print("\n" + "=" * 60)
print("ACTUAL IMAGE FILES")
print("=" * 60)

# Check organized_poultry_dataset
organized_dir = Path("organized_poultry_dataset")
if organized_dir.exists():
    print(f"\n📁 {organized_dir}/")
    total_organized = 0
    for disease_dir in sorted(organized_dir.iterdir()):
        if disease_dir.is_dir():
            image_files = list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.png")) + \
                         list(disease_dir.glob("*.tif")) + list(disease_dir.glob("*.tiff"))
            count = len(image_files)
            total_organized += count
            print(f"   {disease_dir.name:15} {count:5,} images")
    print(f"   {'TOTAL':15} {total_organized:5,} images")

# Check poultry_microscopy
microscopy_dir = Path("poultry_microscopy")
if microscopy_dir.exists():
    print(f"\n📁 {microscopy_dir}/")
    image_files = list(microscopy_dir.rglob("*.jpg")) + list(microscopy_dir.rglob("*.png")) + \
                 list(microscopy_dir.rglob("*.tif")) + list(microscopy_dir.rglob("*.tiff"))
    print(f"   Total images: {len(image_files):,}")

print("\n" + "=" * 60)
