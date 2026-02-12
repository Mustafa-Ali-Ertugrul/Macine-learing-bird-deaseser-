"""
Data Augmentation Script for Poultry Disease Dataset
Hedef: Her sınıfta minimum 500 görüntü olmasını sağlamak
"""

import os
import sys
from pathlib import Path
from PIL import Image
import random
from collections import defaultdict
import shutil

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Dataset path
DATASET_DIR = Path("final_dataset_10_classes")
MIN_IMAGES = 500

def analyze_dataset():
    """Analyze current dataset and identify classes needing augmentation"""
    print("=" * 70)
    print("DATASET ANALİZİ - Minimum 500 Görüntü Hedefi")
    print("=" * 70)
    
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
    print(f"\n📊 Mevcut Sınıf Dağılımı:")
    print("-" * 50)
    
    total = sum(class_counts.values())
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        status = "✅" if count >= MIN_IMAGES else "❌"
        pct = (count / total * 100) if total > 0 else 0
        print(f"{status} {class_name:30} : {count:5} görüntü ({pct:5.1f}%)")
    
    print("-" * 50)
    print(f"Toplam: {total} görüntü")
    
    # Print augmentation needs
    if classes_needing_augmentation:
        print(f"\n⚠️ AUGMENTATION GEREKTİREN SINIFLAR ({len(classes_needing_augmentation)} adet):")
        print("-" * 50)
        print(f"{'Sınıf':<30} {'Mevcut':>8} {'Gerekli':>8} {'Hedef':>8}")
        print("-" * 50)
        
        total_needed = 0
        for class_name, info in sorted(classes_needing_augmentation.items(), key=lambda x: x[1]['needed'], reverse=True):
            print(f"{class_name:<30} {info['current']:>8} {info['needed']:>8} {info['target']:>8}")
            total_needed += info['needed']
        
        print("-" * 50)
        print(f"{'TOPLAM EKSİK':<30} {'':<8} {total_needed:>8}")
    else:
        print("\n✅ Tüm sınıflar minimum 500 görüntü hedefini karşılıyor!")
    
    return class_counts, classes_needing_augmentation

def apply_augmentation(image):
    """Apply random augmentation to an image"""
    augmentations = []
    
    # Random horizontal flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmentations.append("hflip")
    
    # Random vertical flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        augmentations.append("vflip")
    
    # Random rotation (90, 180, 270 degrees)
    rotation = random.choice([0, 90, 180, 270])
    if rotation > 0:
        image = image.rotate(rotation, expand=True)
        augmentations.append(f"rot{rotation}")
    
    # Random brightness adjustment
    if random.random() > 0.5:
        from PIL import ImageEnhance
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        augmentations.append("bright")
    
    # Random contrast adjustment
    if random.random() > 0.5:
        from PIL import ImageEnhance
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
        augmentations.append("contrast")
    
    return image, "_".join(augmentations) if augmentations else "copy"

def augment_class(class_name, current_count, needed_count):
    """Augment images for a specific class"""
    class_dir = DATASET_DIR / class_name
    
    # Get all valid images
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    original_images = [img for img in class_dir.glob("*.*") 
                       if img.suffix.lower() in valid_extensions 
                       and not img.stem.startswith("aug_")]
    
    if not original_images:
        print(f"  ❌ {class_name}: Orijinal görüntü bulunamadı!")
        return 0
    
    print(f"\n🔄 {class_name} sınıfı için augmentation başlatılıyor...")
    print(f"   Orijinal görüntü sayısı: {len(original_images)}")
    print(f"   Hedef ek görüntü: {needed_count}")
    
    created_count = 0
    aug_index = 0
    
    while created_count < needed_count:
        # Select a random original image
        source_image_path = random.choice(original_images)
        
        try:
            with Image.open(source_image_path) as img:
                img = img.convert('RGB')
                augmented_img, aug_type = apply_augmentation(img)
                
                # Generate unique filename
                new_filename = f"aug_{aug_index}_{source_image_path.stem}.jpg"
                new_path = class_dir / new_filename
                
                # Ensure unique filename
                while new_path.exists():
                    aug_index += 1
                    new_filename = f"aug_{aug_index}_{source_image_path.stem}.jpg"
                    new_path = class_dir / new_filename
                
                # Save augmented image
                augmented_img.save(new_path, 'JPEG', quality=95)
                created_count += 1
                aug_index += 1
                
                if created_count % 50 == 0:
                    print(f"   İlerleme: {created_count}/{needed_count} görüntü oluşturuldu")
                    
        except Exception as e:
            print(f"   ⚠️ Hata ({source_image_path.name}): {e}")
            continue
    
    print(f"   ✅ {created_count} yeni görüntü oluşturuldu")
    return created_count

def main():
    print("🐔 Kanatlı Hastalıkları Veri Seti Augmentation Aracı")
    print("=" * 70)
    
    if not DATASET_DIR.exists():
        print(f"❌ Dataset dizini bulunamadı: {DATASET_DIR}")
        return
    
    # Analyze current state
    class_counts, classes_needing_augmentation = analyze_dataset()
    
    if not classes_needing_augmentation:
        print("\n✅ Augmentation gerekmiyor!")
        return
    
    # Ask for confirmation
    print(f"\n" + "=" * 70)
    print("AUGMENTATION PLANI")
    print("=" * 70)
    
    total_to_create = sum(info['needed'] for info in classes_needing_augmentation.values())
    print(f"Toplam {len(classes_needing_augmentation)} sınıf için {total_to_create} yeni görüntü oluşturulacak.")
    
    response = input("\nDevam etmek istiyor musunuz? (e/h): ").strip().lower()
    
    if response != 'e':
        print("İşlem iptal edildi.")
        return
    
    # Perform augmentation
    print("\n" + "=" * 70)
    print("AUGMENTATION BAŞLATILIYOR")
    print("=" * 70)
    
    total_created = 0
    for class_name, info in classes_needing_augmentation.items():
        created = augment_class(class_name, info['current'], info['needed'])
        total_created += created
    
    # Final report
    print("\n" + "=" * 70)
    print("AUGMENTATION TAMAMLANDI")
    print("=" * 70)
    print(f"Toplam oluşturulan görüntü: {total_created}")
    
    # Verify final counts
    print("\n📊 Güncel Sınıf Dağılımı:")
    analyze_dataset()

if __name__ == "__main__":
    main()
