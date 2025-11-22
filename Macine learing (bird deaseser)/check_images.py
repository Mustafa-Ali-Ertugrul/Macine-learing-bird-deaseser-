from pathlib import Path
from PIL import Image
import os

print("=" * 60)
print("IMAGE VERIFICATION")
print("=" * 60)

# Count images
microscopy_dir = Path("poultry_microscopy")
organized_dir = Path("organized_poultry_dataset")

if microscopy_dir.exists():
    jpg_files = list(microscopy_dir.rglob("*.jpg"))
    png_files = list(microscopy_dir.rglob("*.png"))
    tif_files = list(microscopy_dir.rglob("*.tif"))
    tiff_files = list(microscopy_dir.rglob("*.tiff"))
    
    total = len(jpg_files) + len(png_files) + len(tif_files) + len(tiff_files)
    
    print(f"\n📁 poultry_microscopy/")
    print(f"   Total images: {total:,}")
    print(f"   - JPG: {len(jpg_files):,}")
    print(f"   - PNG: {len(png_files):,}")
    print(f"   - TIF/TIFF: {len(tif_files) + len(tiff_files):,}")
    
    # Check for disease-specific folders
    print(f"\n🔬 Disease-specific images:")
    disease_keywords = ['ncd', 'newcastle', 'marek', 'influenza', 'salmo', 'cocci', 'healthy']
    for keyword in disease_keywords:
        count = len([f for f in jpg_files + png_files + tif_files if keyword.lower() in f.name.lower()])
        if count > 0:
            print(f"   {keyword}: {count:,} images")
    
    # Sample images
    print(f"\n📸 Sample images (first 5):")
    all_images = jpg_files + png_files + tif_files
    for img in all_images[:5]:
        size_mb = img.stat().st_size / (1024 * 1024)
        print(f"   {img.name} - {size_mb:.2f} MB")

if organized_dir.exists():
    print(f"\n📁 organized_poultry_dataset/")
    for disease_folder in sorted(organized_dir.iterdir()):
        if disease_folder.is_dir():
            images = list(disease_folder.glob("*.jpg")) + list(disease_folder.glob("*.png")) + \
                    list(disease_folder.glob("*.tif")) + list(disease_folder.glob("*.tiff"))
            print(f"   {disease_folder.name}: {len(images):,} images")

# Verify image quality (sample check)
print(f"\n✅ Image Quality Check (5 random samples):")
sample_images = all_images[:5]
for img_path in sample_images:
    try:
        with Image.open(img_path) as img:
            print(f"   {img_path.name}: {img.size[0]}x{img.size[1]} - {img.mode} - OK")
    except Exception as e:
        print(f"   {img_path.name}: ERROR - {str(e)[:50]}")

print("\n" + "=" * 60)
