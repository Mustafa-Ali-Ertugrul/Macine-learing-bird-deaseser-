"""
Automatic Data Augmentation Script for Poultry Disease Dataset
Hedef: Her sınıfta minimum 500 görüntü olmasını sağlamak
Kullanıcı onayı olmadan otomatik çalışır
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
import random

# Dataset path
DATASET_DIR = Path("final_dataset_10_classes")
MIN_IMAGES = 500

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
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        augmentations.append("bright")
    
    # Random contrast adjustment
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
        augmentations.append("contrast")
    
    # Random saturation adjustment
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(factor)
        augmentations.append("saturation")
    
    return image, "_".join(augmentations) if augmentations else "copy"

def augment_class(class_name, current_count, needed_count):
    """Augment images for a specific class"""
    class_dir = DATASET_DIR / class_name
    
    # Get all valid images (excluding already augmented ones)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    original_images = [img for img in class_dir.glob("*.*") 
                       if img.suffix.lower() in valid_extensions 
                       and not img.stem.startswith("aug_")]
    
    if not original_images:
        print(f"  [HATA] {class_name}: Orijinal goruntu bulunamadi!")
        return 0
    
    print(f"\n[ISLEM] {class_name} sinifi icin augmentation baslatiliyor...")
    print(f"   Orijinal goruntu sayisi: {len(original_images)}")
    print(f"   Hedef ek goruntu: {needed_count}")
    
    created_count = 0
    aug_index = 0
    max_attempts = needed_count * 3  # Prevent infinite loops
    attempts = 0
    
    while created_count < needed_count and attempts < max_attempts:
        attempts += 1
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
                    print(f"   Ilerleme: {created_count}/{needed_count} goruntu olusturuldu")
                    
        except Exception as e:
            print(f"   [UYARI] Hata ({source_image_path.name}): {e}")
            continue
    
    print(f"   [TAMAM] {created_count} yeni goruntu olusturuldu")
    return created_count

def main():
    print("=" * 70)
    print("KANATLI HASTALIKLARI VERI SETI AUGMENTATION ARACI")
    print("Hedef: Her sinifta minimum 500 goruntu")
    print("=" * 70)
    
    if not DATASET_DIR.exists():
        print(f"[HATA] Dataset dizini bulunamadi: {DATASET_DIR}")
        return
    
    # Analyze current state
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
    print("\n[ANALIZ] Mevcut Sinif Dagilimi:")
    print("-" * 50)
    
    total = sum(class_counts.values())
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        status = "[OK]" if count >= MIN_IMAGES else "[EKSIK]"
        print(f"{status:8} {class_name:30} : {count:5} goruntu")
    
    print("-" * 50)
    print(f"Toplam: {total} goruntu")
    
    if not classes_needing_augmentation:
        print("\n[TAMAM] Tum siniflar minimum 500 goruntu hedefini karsiliyor!")
        return
    
    # Print augmentation plan
    print(f"\n[PLAN] Augmentation gerektiren siniflar ({len(classes_needing_augmentation)} adet):")
    print("-" * 60)
    
    total_needed = sum(info['needed'] for info in classes_needing_augmentation.values())
    for class_name, info in sorted(classes_needing_augmentation.items(), key=lambda x: x[1]['needed'], reverse=True):
        print(f"   {class_name}: {info['current']} -> {info['target']} (+{info['needed']})")
    
    print(f"\nToplam olusturulacak goruntu: {total_needed}")
    
    # Perform augmentation
    print("\n" + "=" * 70)
    print("AUGMENTATION BASLATILIYOR")
    print("=" * 70)
    
    total_created = 0
    for class_name, info in classes_needing_augmentation.items():
        created = augment_class(class_name, info['current'], info['needed'])
        total_created += created
    
    # Final report
    print("\n" + "=" * 70)
    print("AUGMENTATION TAMAMLANDI")
    print("=" * 70)
    print(f"Toplam olusturulan goruntu: {total_created}")
    
    # Verify final counts
    print("\n[DOGRULAMA] Guncel Sinif Dagilimi:")
    print("-" * 50)
    
    new_total = 0
    for class_dir in sorted(DATASET_DIR.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.*"))
            valid_images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]
            count = len(valid_images)
            new_total += count
            status = "[OK]" if count >= MIN_IMAGES else "[EKSIK]"
            print(f"{status:8} {class_dir.name:30} : {count:5} goruntu")
    
    print("-" * 50)
    print(f"Yeni Toplam: {new_total} goruntu (Eklenen: {new_total - total})")

if __name__ == "__main__":
    main()
