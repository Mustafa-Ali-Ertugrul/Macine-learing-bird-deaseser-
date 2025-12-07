
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

# Mapping label from CSV to Folder Names
LABEL_TO_FOLDER = {
    'coccidiosis': 'Coccidiosis',
    'healthy': 'Healthy',
    'ncd': 'Newcastle Disease',
    'pcrcocci': 'Coccidiosis',
    'pcrhealthy': 'Healthy', 
    'pcrncd': 'Newcastle Disease',
    'pcrsalmo': 'Salmonella',
    'salmonella': 'Salmonella',
    # Note: Avian Influenza, Fowl Pox etc are NOT in the CSV, assuming they are already in folders
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
    print("🚀 Starting Dataset Balancing...")
    
    # 1. Distribute images from CSV if not already in folders
    # The folders already have data (from previous steps potentially), but we need to ensure everything usable is there.
    # However, the user asked to check datasets again. 
    # If I just augment existing folders, I might miss the data in CSV?
    # Let's check if the CSV images are ALREADY in the specific folders?
    # The CSV paths are like `poultry_microscopy/...` while target folders are `final_dataset_10_classes/...`
    # They are likely different copies. 
    # I will copy VALID images from CSV to the target folders to maximize raw data before augmentation.
    
    if os.path.exists(SOURCE_CSV):
        df = pd.read_csv(SOURCE_CSV)
        print(f"Scanning {len(df)} rows from CSV to consolidate data...")
        
        for idx, row in df.iterrows():
            label = str(row['disease']).lower()
            if label in LABEL_TO_FOLDER:
                folder_name = LABEL_TO_FOLDER[label]
                # Find the destination folder that matches this name partially
                # (e.g. 'Coccidiosis' matches 'Coccidiosis - Bukoola Vet' ?)
                # The user folders are specific. Let's find exact or partial match.
                
                target_folder = None
                for d in os.listdir(TARGET_DIR):
                    if folder_name.lower() in d.lower():
                        target_folder = os.path.join(TARGET_DIR, d)
                        break
                
                if target_folder:
                    # Construct source path
                    rel_path = row['image_path']
                    if pd.isna(rel_path): continue
                    rel_path = rel_path.replace('\\', '/')
                    src_path = os.path.join(IMAGE_BASE_DIR, rel_path)
                    
                    if os.path.exists(src_path):
                        # Destination filename (avoid overwrite collision)
                        filename = os.path.basename(src_path)
                        dst_path = os.path.join(target_folder, f"csv_{idx}_{filename}")
                        
                        if not os.path.exists(dst_path):
                            try:
                                shutil.copy(src_path, dst_path)
                            except:
                                pass
    
    # 2. Check counts and Augment
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
                src_full = os.path.join(cls_path, src_name)
                
                # New name
                new_name = f"aug_{generated}_{src_name}"
                dst_full = os.path.join(cls_path, new_name)
                
                if augment_image(src_full, dst_full):
                    generated += 1
                    if generated % 100 == 0:
                        print(f"      Generated {generated}...")
            
            print(f"   ✅ Augmented {generated} images. New Total: {count + generated}")
        else:
            print("   ✅ Count OK.")

    print("\n🎉 Balancing Complete.")

if __name__ == "__main__":
    main()
