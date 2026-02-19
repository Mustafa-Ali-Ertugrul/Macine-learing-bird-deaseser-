import pandas as pd
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("COMPREHENSIVE POULTRY DISEASE DATASET ANALYSIS")
print("=" * 70)

# Analyze CSV metadata
csv_file = "poultry_labeled_12k.csv"
if Path(csv_file).exists():
    df = pd.read_csv(csv_file)
    
    print(f"\n📊 CSV Metadata Analysis ({csv_file}):")
    print(f"   Total records: {len(df):,}")
    
    # Disease distribution from CSV
    disease_counts = df['disease'].value_counts()
    
    print(f"\n📈 Disease Distribution in CSV:")
    print(f"   {'Category':<20} {'Count':>8}  {'Percentage':>10}")
    print(f"   {'-'*20} {'-'*8}  {'-'*10}")
    
    for disease, count in disease_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {disease:<20} {count:>8,}  {pct:>9.1f}%")
    
    print(f"   {'-'*20} {'-'*8}  {'-'*10}")
    print(f"   {'TOTAL':<20} {len(df):>8,}  {100.0:>9.1f}%")

# Analyze actual image files
print("\n" + "=" * 70)
print("ACTUAL IMAGE FILES ON DISK")
print("=" * 70)

microscopy_dir = Path("poultry_microscopy")
if microscopy_dir.exists():
    # Count all images
    all_images = list(microscopy_dir.rglob("*.jpg")) + \
                 list(microscopy_dir.rglob("*.png")) + \
                 list(microscopy_dir.rglob("*.tif")) + \
                 list(microscopy_dir.rglob("*.tiff"))
    
    print(f"\n📁 poultry_microscopy/")
    print(f"   Total images: {len(all_images):,}")
    
    # Categorize by filename patterns
    disease_images = defaultdict(int)
    
    for img in all_images:
        name_lower = img.name.lower()
        
        # Check for disease keywords in filename
        if 'pcrncd' in name_lower or 'ncd' in name_lower:
            disease_images['Newcastle (NCD)'] += 1
        elif 'pcrsalmo' in name_lower or 'salmo' in name_lower:
            disease_images['Salmonella'] += 1
        elif 'pcrcocci' in name_lower or 'cocci' in name_lower:
            disease_images['Coccidiosis'] += 1
        elif 'pcrhealthy' in name_lower or 'healthy' in name_lower or 'normal' in name_lower:
            disease_images['Healthy'] += 1
        elif 'marek' in name_lower or 'mdv' in name_lower:
            disease_images['Marek'] += 1
        elif 'influenza' in name_lower or 'flu' in name_lower or 'h5n1' in name_lower:
            disease_images['Avian Influenza'] += 1
        elif 'ib' in name_lower and 'ibd' not in name_lower:
            disease_images['IB'] += 1
        elif 'ibd' in name_lower:
            disease_images['IBD'] += 1
        elif 'fatty' in name_lower or 'liver' in name_lower:
            disease_images['Fatty Liver'] += 1
        elif 'histomon' in name_lower:
            disease_images['Histomoniasis'] += 1
        else:
            disease_images['Uncategorized'] += 1
    
    print(f"\n   Disease breakdown by filename:")
    print(f"   {'Category':<25} {'Count':>8}  {'Percentage':>10}")
    print(f"   {'-'*25} {'-'*8}  {'-'*10}")
    
    total_images = len(all_images)
    for disease in sorted(disease_images.keys()):
        count = disease_images[disease]
        pct = (count / total_images) * 100
        print(f"   {disease:<25} {count:>8,}  {pct:>9.1f}%")
    
    print(f"   {'-'*25} {'-'*8}  {'-'*10}")
    print(f"   {'TOTAL':<25} {total_images:>8,}  {100.0:>9.1f}%")

# Organized dataset
organized_dir = Path("organized_poultry_dataset")
if organized_dir.exists():
    print(f"\n📁 organized_poultry_dataset/")
    
    total_organized = 0
    for disease_folder in sorted(organized_dir.iterdir()):
        if disease_folder.is_dir():
            images = list(disease_folder.glob("*.jpg")) + \
                    list(disease_folder.glob("*.png")) + \
                    list(disease_folder.glob("*.tif")) + \
                    list(disease_folder.glob("*.tiff"))
            count = len(images)
            total_organized += count
            print(f"   {disease_folder.name:<20} {count:>8,} images")
    
    print(f"   {'-'*20} {'-'*8}")
    print(f"   {'TOTAL':<20} {total_organized:>8,} images")

# Grand total
print("\n" + "=" * 70)
print("GRAND TOTAL SUMMARY")
print("=" * 70)
print(f"\n🎯 Total unique images in dataset: {total_images:,}")
print(f"📊 CSV metadata records: {len(df):,}")
print(f"📁 Organized (ready for training): {total_organized:,}")
print(f"🔬 Raw images (poultry_microscopy): {total_images:,}")

print("\n" + "=" * 70)
