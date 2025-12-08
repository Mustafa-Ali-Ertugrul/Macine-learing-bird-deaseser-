import os
import sys
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Dataset path
dataset_dir = Path("final_dataset_10_classes")

print("=" * 60)
print("POULTRY DATASET VERIFICATION")
print("=" * 60)
print(f"Checking directory: {dataset_dir.absolute()}")

if not dataset_dir.exists():
    print(f"❌ Error: Directory not found!")
    sys.exit(1)

# Initialize stats
class_stats = defaultdict(int)
corrupted_files = []
non_image_files = []
total_files = 0
valid_images = 0

# Walk through the dataset
for class_dir in sorted(dataset_dir.iterdir()):
    if class_dir.is_dir():
        class_name = class_dir.name
        print(f"\n📁 Class: {class_name}")
        
        # files in this class
        files = list(class_dir.iterdir())
        class_files = 0
        
        for file_path in tqdm(files, desc=f"Checking {class_name}", unit="img"):
            if file_path.is_file():
                total_files += 1
                
                # Check extension
                if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
                    non_image_files.append(str(file_path))
                    continue
                
                # Verify image integrity
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify file integrity
                    
                    # Re-open to check if it can be loaded (verify doesn't load)
                    # Some files pass verify but fail load/transposed
                    with Image.open(file_path) as img:
                        img.load()
                        
                    class_stats[class_name] += 1
                    class_files += 1
                    valid_images += 1
                except Exception as e:
                    print(f"   ❌ Corrupted: {file_path.name} - {str(e)}")
                    corrupted_files.append((str(file_path), str(e)))

        print(f"   ✅ Valid images: {class_files}")

print("\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)
print(f"Total files scanned: {total_files}")
print(f"Total valid images: {valid_images}")

print("\n📊 Class Distribution:")
for class_name, count in sorted(class_stats.items()):
    percentage = (count / valid_images * 100) if valid_images > 0 else 0
    print(f"   {class_name:25} : {count:5} images ({percentage:5.1f}%)")

if corrupted_files:
    print(f"\n❌ FOUND {len(corrupted_files)} CORRUPTED IMAGES:")
    for path, error in corrupted_files:
        print(f"   - {os.path.basename(path)}: {error}")
else:
    print("\n✅ No corrupted images found.")

if non_image_files:
    print(f"\n⚠️ FOUND {len(non_image_files)} NON-IMAGE FILES:")
    for path in non_image_files:
        print(f"   - {os.path.basename(path)}")

print("=" * 60)
