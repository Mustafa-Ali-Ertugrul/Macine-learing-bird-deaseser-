
import os
import shutil
import pandas as pd
from PIL import Image, ImageEnhance
import random

# === Configuration ===
SOURCE_CSV = 'Macine learing (bird deaseser)/poultry_labeled_12k.csv'
IMAGE_BASE_DIR = 'data/processed/poultry_dataset_512x512'
TARGET_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
MIN_IMAGES_PER_CLASS = 500

# Mapping from CSV labels to Standardized Folder Names
LABEL_TO_FOLDER = {
    'coccidiosis': 'Coccidiosis',
    'healthy': 'Healthy',
    'ncd': 'Newcastle_Disease',
    'pcrcocci': 'Coccidiosis',
    'pcrhealthy': 'Healthy', 
    'pcrncd': 'Newcastle_Disease',
    'pcrsalmo': 'Salmonella',
    'salmonella': 'Salmonella',
    # Note: Other classes (Avian Influenza etc.) are not in this CSV, so they are not reset strictly by this,
    # but we will check strict counts for them during augmentation phase.
}

def augment_image(image_path, save_path):
    """
    Apply random augmentation (flip, rotate, brightness) and save.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Random operations
        op = random.choice(['flip', 'rotate', 'brightness'])
        
        if op == 'flip':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == 'rotate':
            angle = random.choice([90, 180, 270])
            img = img.rotate(angle)
        elif op == 'brightness':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            
        img.save(save_path, quality=95)
        return True
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")
        return False

def main():
    print("🚀 Starting Dataset Restoration (From CSV)...")
    
    # 1. Consolidate from CSV (Restoring images)
    if os.path.exists(SOURCE_CSV):
        print(f"Scanning {SOURCE_CSV} to restore original images...")
        df = pd.read_csv(SOURCE_CSV)
        
        count_moved = 0
        for _, row in df.iterrows():
            label = str(row['disease']).lower()
            rel_path = row['image_path']
            
            if pd.isna(rel_path) or label not in LABEL_TO_FOLDER:
                continue
                
            target_class = LABEL_TO_FOLDER[label]
            target_folder = os.path.join(TARGET_DIR, target_class)
            
            # Ensure target folder exists
            os.makedirs(target_folder, exist_ok=True)
            
            # Source path
            src_path = os.path.join(IMAGE_BASE_DIR, rel_path.replace('\\', '/'))
            
            if os.path.exists(src_path):
                # Destination filename
                filename = os.path.basename(src_path)
                dst_path = os.path.join(target_folder, filename)
                
                # Copy if not exists (Restore!)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    count_moved += 1
                    if count_moved % 100 == 0:
                        print(f"Restored {count_moved} images...", end='\r')
        print(f"\n✅ Restored {count_moved} images from CSV.")
    else:
        print(f"⚠️ Source CSV {SOURCE_CSV} not found. Skipping restoration.")

    # 2. Check counts and Augment (Ensure minimum 500)
    print("\n📊 Checking Class Counts and Augmenting...")
    
    for cls in sorted(os.listdir(TARGET_DIR)):
        cls_path = os.path.join(TARGET_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
            
        # Get valid images
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        print(f"Class: {cls} | Count: {count}")
        
        if count < MIN_IMAGES_PER_CLASS:
            needed = MIN_IMAGES_PER_CLASS - count
            print(f"   ⚠️ Needs {needed} more images. Augmenting...")
            
            if count == 0:
                print(f"   ❌ Error: Class {cls} has 0 images. Cannot augment.")
                continue
                
            # Augment
            generated = 0
            while generated < needed:
                # Pick a random source image
                src_name = random.choice(images)
                # Avoid augmenting already augmented images if possible for variety, 
                # but for simplicity just pick from whatever is there.
                src_full = os.path.join(cls_path, src_name)
                
                # New name
                new_name = f"aug_restore_{generated}_{src_name}"
                dst_full = os.path.join(cls_path, new_name)
                
                if augment_image(src_full, dst_full):
                    generated += 1
                    if generated % 100 == 0:
                        print(f"      Generated {generated}...")
            
            print(f"   ✅ Augmented {generated} images. New Total: {count + generated}")
        else:
            print(f"   ✅ Count OK ({count} >= {MIN_IMAGES_PER_CLASS}).")

    print("\n🎉 Restoration & Balancing Complete.")

if __name__ == "__main__":
    main()
