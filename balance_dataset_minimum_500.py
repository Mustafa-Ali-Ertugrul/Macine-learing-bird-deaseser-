# balance_dataset_minimum_500.py
"""
Dataset Dengeleme Script'i
Her sınıfın en az 500 görüntüsü olacak şekilde dataset'i dengeler.
- Küçük sınıflar için augmentasyon uygular
- Val/Test setlerinin de dengeli olmasını sağlar
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(r"c:\Users\Ali\OneDrive\Belgeler\pyton\Macine learing (bird deaseser)\final_dataset_split")
MIN_IMAGES_PER_CLASS = 500
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Augmentation functions
def augment_image(img_path, output_path, aug_type):
    """Apply augmentation to an image"""
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if aug_type == 'flip_h':
            img = ImageOps.mirror(img)
        elif aug_type == 'flip_v':
            img = ImageOps.flip(img)
        elif aug_type == 'rotate_90':
            img = img.rotate(90, expand=True)
        elif aug_type == 'rotate_180':
            img = img.rotate(180)
        elif aug_type == 'rotate_270':
            img = img.rotate(270, expand=True)
        elif aug_type == 'brightness_up':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.3)
        elif aug_type == 'brightness_down':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.7)
        elif aug_type == 'contrast_up':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
        elif aug_type == 'contrast_down':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.7)
        elif aug_type == 'color_enhance':
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
        
        img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"  [HATA] {img_path}: {e}")
        return False

def get_class_counts(base_dir):
    """Get image counts for each class in each split"""
    splits = ['train', 'val', 'test']
    counts = defaultdict(lambda: defaultdict(int))
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                image_files = [f for f in class_dir.iterdir() 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
                counts[class_dir.name][split] = len(image_files)
    
    return counts

def get_all_images_for_class(base_dir, class_name):
    """Get all image paths for a class across all splits"""
    images = []
    for split in ['train', 'val', 'test']:
        class_dir = base_dir / split / class_name
        if class_dir.exists():
            for f in class_dir.iterdir():
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    images.append((f, split))
    return images

def balance_dataset():
    """Main function to balance the dataset"""
    print("=" * 70)
    print("DATASET DENGELEME - Minimum 500 görüntü/sınıf")
    print("=" * 70)
    
    # Get current counts
    print("\n📊 Mevcut durum analiz ediliyor...")
    counts = get_class_counts(BASE_DIR)
    
    classes = sorted(counts.keys())
    print(f"\n{'Sınıf':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Toplam':>10}")
    print("-" * 70)
    
    for cls in classes:
        train = counts[cls]['train']
        val = counts[cls]['val']
        test = counts[cls]['test']
        total = train + val + test
        status = "✅" if total >= MIN_IMAGES_PER_CLASS else "⚠️"
        print(f"{cls:<30} {train:>8} {val:>8} {test:>8} {total:>10} {status}")
    
    # Process each class
    print("\n" + "=" * 70)
    print("🔧 Dengeleme işlemi başlıyor...")
    print("=" * 70)
    
    aug_types = ['flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270',
                 'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down', 'color_enhance']
    
    for cls in classes:
        train = counts[cls]['train']
        val = counts[cls]['val']
        test = counts[cls]['test']
        total = train + val + test
        
        if total >= MIN_IMAGES_PER_CLASS:
            # Check if val/test have enough samples
            min_val_test = int(MIN_IMAGES_PER_CLASS * 0.15)  # At least 15% for val and test
            if val < min_val_test or test < min_val_test:
                print(f"\n📂 {cls}: Val/Test dengesiz, yeniden dağıtım yapılacak...")
                redistribute_splits(BASE_DIR, cls, min_val_test)
            else:
                print(f"\n✅ {cls}: Yeterli ({total} görüntü)")
            continue
        
        needed = MIN_IMAGES_PER_CLASS - total
        print(f"\n🔄 {cls}: {total} → {MIN_IMAGES_PER_CLASS} (+{needed} gerekli)")
        
        # Get all existing images for this class
        all_images = get_all_images_for_class(BASE_DIR, cls)
        
        if not all_images:
            print(f"  [UYARI] Bu sınıf için görüntü bulunamadı!")
            continue
        
        # Create augmented images in train folder
        train_dir = BASE_DIR / 'train' / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        
        aug_count = 0
        aug_idx = 0
        
        while aug_count < needed:
            # Choose random source image (prefer train images)
            train_images = [img for img, split in all_images if split == 'train']
            if not train_images:
                train_images = [img for img, split in all_images]
            
            src_img = random.choice(train_images)
            aug_type = aug_types[aug_idx % len(aug_types)]
            
            # Create unique filename
            base_name = src_img.stem
            new_name = f"aug_{aug_idx:04d}_{aug_type}_{base_name}.jpg"
            output_path = train_dir / new_name
            
            if augment_image(src_img, output_path, aug_type):
                aug_count += 1
            
            aug_idx += 1
            
            # Safety: prevent infinite loop
            if aug_idx > needed * 3:
                print(f"  [UYARI] Maksimum deneme sayısına ulaşıldı")
                break
        
        print(f"  ✅ {aug_count} augmente görüntü oluşturuldu")
        
        # Redistribute to ensure val/test have enough
        min_val_test = int(MIN_IMAGES_PER_CLASS * 0.15)
        redistribute_splits(BASE_DIR, cls, min_val_test)
    
    # Final report
    print("\n" + "=" * 70)
    print("📊 SONUÇ RAPORU")
    print("=" * 70)
    
    final_counts = get_class_counts(BASE_DIR)
    total_images = 0
    
    print(f"\n{'Sınıf':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Toplam':>10}")
    print("-" * 70)
    
    for cls in sorted(final_counts.keys()):
        train = final_counts[cls]['train']
        val = final_counts[cls]['val']
        test = final_counts[cls]['test']
        total = train + val + test
        total_images += total
        status = "✅" if total >= MIN_IMAGES_PER_CLASS else "❌"
        print(f"{cls:<30} {train:>8} {val:>8} {test:>8} {total:>10} {status}")
    
    print("-" * 70)
    print(f"{'TOPLAM':<30} {sum(final_counts[c]['train'] for c in final_counts):>8} "
          f"{sum(final_counts[c]['val'] for c in final_counts):>8} "
          f"{sum(final_counts[c]['test'] for c in final_counts):>8} {total_images:>10}")
    
    print("\n✅ Dataset dengeleme tamamlandı!")

def redistribute_splits(base_dir, class_name, min_per_split):
    """Redistribute images to ensure val/test have minimum samples"""
    train_dir = base_dir / 'train' / class_name
    val_dir = base_dir / 'val' / class_name
    test_dir = base_dir / 'test' / class_name
    
    # Ensure directories exist
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_images = []
    for split, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        if split_dir.exists():
            for f in split_dir.iterdir():
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    all_images.append(f)
    
    if len(all_images) < MIN_IMAGES_PER_CLASS:
        return  # Not enough images to redistribute
    
    # Count current val/test
    val_count = len(list(val_dir.glob('*.*'))) if val_dir.exists() else 0
    test_count = len(list(test_dir.glob('*.*'))) if test_dir.exists() else 0
    
    # Move from train to val if needed
    if val_count < min_per_split:
        needed = min_per_split - val_count
        train_images = [f for f in train_dir.iterdir() 
                       if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
        # Prefer non-augmented images for val/test
        non_aug = [f for f in train_images if not f.name.startswith('aug_')]
        aug = [f for f in train_images if f.name.startswith('aug_')]
        source_images = non_aug + aug
        
        for i, img in enumerate(source_images[:needed]):
            shutil.move(str(img), str(val_dir / img.name))
        print(f"  → {min(needed, len(source_images))} görüntü val'e taşındı")
    
    # Move from train to test if needed
    if test_count < min_per_split:
        needed = min_per_split - test_count
        train_images = [f for f in train_dir.iterdir() 
                       if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
        non_aug = [f for f in train_images if not f.name.startswith('aug_')]
        aug = [f for f in train_images if f.name.startswith('aug_')]
        source_images = non_aug + aug
        
        for i, img in enumerate(source_images[:needed]):
            shutil.move(str(img), str(test_dir / img.name))
        print(f"  → {min(needed, len(source_images))} görüntü test'e taşındı")

if __name__ == "__main__":
    balance_dataset()
