
import os
import shutil
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance
import numpy as np

# Config
SOURCE_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
TARGET_BASE = 'Macine learing (bird deaseser)/final_dataset_split'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test ratio is remainder (0.1)

MIN_TRAIN_SAMPLES = 200 # Augment up to this

def ensure_clean_dir(path):
    if os.path.exists(path):
        current_time = str(int(os.path.getmtime(path)))
        # rename old
        shutil.move(path, f"{path}_backup_{current_time}")
    os.makedirs(path)

def augment_image(image_path, save_path):
    try:
        img = Image.open(image_path).convert('RGB')
        op = random.choice(['flip', 'rotate', 'brightness', 'contrast'])
        
        if op == 'flip':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == 'rotate':
            angle = random.choice([90, 180, 270])
            img = img.rotate(angle)
        elif op == 'brightness':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        elif op == 'contrast':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            
        img.save(save_path, quality=95)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def organize_dataset():
    print("🚀 Starting Physical Dataset Split...")
    
    # 1. Setup Dirs
    for split in ['train', 'val', 'test']:
        p = os.path.join(TARGET_BASE, split)
        if os.path.exists(p):
            shutil.rmtree(p) # Clean start
        os.makedirs(p)
        
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    for cls in classes:
        print(f"\nProcessing {cls}...")
        cls_src = os.path.join(SOURCE_DIR, cls)
        
        # 2. Get Clean Images (Exclude previous safe_aug)
        all_files = os.listdir(cls_src)
        clean_files = [f for f in all_files 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg')) 
                      and not f.startswith('safe_aug_')]
        
        # Sort for reproducibility
        clean_files.sort()
        
        # Shuffle deterministically
        random.seed(42)
        random.shuffle(clean_files)
        
        # 3. Calculate Splits
        n = len(clean_files)
        if n == 0:
            print(f"   ⚠️ Skipping {cls} (No clean images)")
            continue
            
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        train_files = clean_files[:n_train]
        val_files = clean_files[n_train:n_train+n_val]
        test_files = clean_files[n_train+n_val:]
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        # 4. Copy Files
        for split_name, files in splits.items():
            dest_dir = os.path.join(TARGET_BASE, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            
            for f in files:
                shutil.copy2(os.path.join(cls_src, f), os.path.join(dest_dir, f))
                
        print(f"   Split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")
        
        # 5. Augment Training Set ONLY
        train_dst = os.path.join(TARGET_BASE, 'train', cls)
        curr_train = len(train_files)
        
        if curr_train < MIN_TRAIN_SAMPLES:
            needed = MIN_TRAIN_SAMPLES - curr_train
            print(f"   🛠️ Augmenting Train ({curr_train} -> {MIN_TRAIN_SAMPLES})...")
            
            generated = 0
            pbar = tqdm(total=needed, leave=False)
            
            while generated < needed:
                src = random.choice(train_files)
                src_path = os.path.join(train_dst, src)
                new_name = f"aug_split_{generated}_{src}"
                dst_path = os.path.join(train_dst, new_name)
                
                if augment_image(src_path, dst_path):
                    generated += 1
                    pbar.update(1)
            pbar.close()

    print("\n✅ Dataset reorganized successfully at:")
    print(f"   {TARGET_BASE}")

if __name__ == "__main__":
    organize_dataset()
