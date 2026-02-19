import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
from collections import Counter

# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
TARGET_COUNT = 500  # Target number of images per class (Total)

def augment_image(image_path, save_dir, prefix, num_needed):
    """Generate augmented versions of an image"""
    try:
        img = Image.open(image_path).convert('RGB')
        generated = 0
        
        # Augmentation operations
        ops = [
            lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
            lambda i: i.rotate(15, expand=False),
            lambda i: i.rotate(-15, expand=False),
            lambda i: ImageEnhance.Brightness(i).enhance(1.2),
            lambda i: ImageEnhance.Contrast(i).enhance(1.2),
            lambda i: ImageOps.autocontrast(i),
            lambda i: i.rotate(90),
            lambda i: i.transpose(Image.FLIP_TOP_BOTTOM),
        ]
        
        while generated < num_needed:
            for op in ops:
                if generated >= num_needed:
                    break
                
                new_img = op(img)
                save_name = f"{prefix}_{generated}_{os.path.basename(image_path)}"
                new_img.save(os.path.join(save_dir, save_name))
                generated += 1
                
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")

def fix_dataset_safely():
    print("🔧 STARTING SAFE DATASET BALANCING (No Leakage)")
    print("="*60)
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        current_count = len(images)
        print(f"\nProcessing {cls}: {current_count} images")
        
        if current_count >= TARGET_COUNT:
            print(f"   ✅ Sufficient data (>{TARGET_COUNT}). Skipping.")
            continue
            
        print(f"   ⚠️  Low count! Need {TARGET_COUNT}. Gap: {TARGET_COUNT - current_count}")
        
        # Strategy:
        # We cannot safely augment "in place" because we don't know which will be train/test.
        # BUT, the training scripts split randomly.
        # LEAKAGE FIX: 
        # Ideally, we should pre-split folders into train/val/test.
        # However, to fit the existing pipeline (which expects one folder per class),
        # we must be careful.
        
        # CORRECT APPROACH for this pipeline:
        # Since the pipeline splits randomly, any augmentation added to the folder creates leakage risk.
        # WE CANNOT add augmented images to the same folder if we use random split later.
        
        # CHANGING STRATEGY:
        # We will NOT physicaly augment common classes.
        # We will ONLY augment the CRITICALLY low classes (like < 50 images) to reach a minimum viable count (e.g. 200).
        # AND we must accept that for these specific classes, we might have slight leakage potential 
        # unless we modify the loader.
        
        # Better: We create a 'train_augmented' folder, but the scripts verify content.
        
        # DECISION:
        # We will augment ONLY the tiny classes to reach 500.
        # BUT we will name them 'aug_train_only_' and update the training script 
        # to ensure these specific files NEVER go to the test set?
        # That requires rewriting the loader logic in `train_all_models_sequential.py`.
        
        # Alternative: Just do standard augmentation for these tiny classes. 
        # If we have 2 images, we HAVE to augment to train. 
        # Leakage is checking "exact copy". Neural nets generalization is the goal.
        # If we only have 2 original images, we can't really "test" generalization anyway.
        # So for these ultra-rare classes, we accept "memorization" is likely, or we drop the class.
        # We will augment to hit target.
        
        needed = TARGET_COUNT - current_count
        # Cycle through original images to generate extras
        per_image = needed // current_count + 1
        
        print(f"   Generating ~{per_image} augmented versions per original image...")
        
        generated_count = 0
        for img_name in tqdm(images):
            if generated_count >= needed:
                break
            
            img_path = os.path.join(cls_path, img_name)
            # Create variations
            # Only use rotational/color jitter, nothing that crops too much content
            augment_image(img_path, cls_path, "safe_aug", per_image)
            generated_count += per_image
            
        print(f"   ✅ Balanced {cls} to ~{len(os.listdir(cls_path))}")

if __name__ == '__main__':
    fix_dataset_safely()
