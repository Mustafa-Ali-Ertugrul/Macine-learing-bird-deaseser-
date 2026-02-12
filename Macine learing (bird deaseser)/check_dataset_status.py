"""
Dataset Status Check Script
"""
import os
from pathlib import Path

DATASET_DIR = Path("final_dataset_10_classes")
MIN_IMAGES = 500

class_counts = {}
classes_needing_augmentation = {}

for class_dir in sorted(DATASET_DIR.iterdir()):
    if class_dir.is_dir():
        images = list(class_dir.glob("*.*"))
        valid_images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]
        count = len(valid_images)
        class_counts[class_dir.name] = count
        
        if count < MIN_IMAGES:
            needed = MIN_IMAGES - count
            classes_needing_augmentation[class_dir.name] = {
                'current': count,
                'needed': needed,
                'target': MIN_IMAGES
            }

# Print current status
print("=" * 70)
print("DATASET ANALIZI - Minimum 500 Goruntu Hedefi")
print("=" * 70)
print()
print("Mevcut Sinif Dagilimi:")
print("-" * 50)

total = sum(class_counts.values())
for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    status = "[OK]" if count >= MIN_IMAGES else "[EKSIK]"
    pct = (count / total * 100) if total > 0 else 0
    print(f"{status:8} {class_name:30} : {count:5} goruntu ({pct:5.1f}%)")

print("-" * 50)
print(f"Toplam: {total} goruntu")

# Print augmentation needs
if classes_needing_augmentation:
    print()
    print(f"AUGMENTATION GEREKTIREN SINIFLAR ({len(classes_needing_augmentation)} adet):")
    print("-" * 60)
    print(f"{'Sinif':<30} {'Mevcut':>10} {'Gerekli':>10} {'Hedef':>10}")
    print("-" * 60)
    
    total_needed = 0
    for class_name, info in sorted(classes_needing_augmentation.items(), key=lambda x: x[1]['needed'], reverse=True):
        print(f"{class_name:<30} {info['current']:>10} {'+' + str(info['needed']):>10} {info['target']:>10}")
        total_needed += info['needed']
    
    print("-" * 60)
    print(f"{'TOPLAM EKSIK':<30} {'':>10} {'+' + str(total_needed):>10}")
else:
    print()
    print("Tum siniflar minimum 500 goruntu hedefini karsiliyor!")
