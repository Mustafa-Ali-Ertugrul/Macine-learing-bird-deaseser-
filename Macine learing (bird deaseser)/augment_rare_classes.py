
import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm

DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
TARGET_COUNT = 200  # Minimum safe count to avoid "Severe Imbalance" errors and enable splitting

def augment_image(image_path, save_path):
    """
    Apply random augmentation (flip, rotate, brightness) and save.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Random operations
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
        print(f"Error augmenting {image_path}: {e}")
        return False

def balance_dataset():
    print(f"Starting targeted augmentation on: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        count = len(images)
        
        print(f"Class '{cls}': {count} images")
        
        if count < TARGET_COUNT:
            needed = TARGET_COUNT - count
            print(f"   ⚠️ Needs {needed} more images (Target: {TARGET_COUNT})")
            
            pbar = tqdm(total=needed, desc=f"Augmenting {cls}", leave=False)
            generated = 0
            
            # Avoid infinite loop if no images
            if count == 0:
                print("      ❌ No source images to augment!")
                continue
                
            while generated < needed:
                # Pick random source
                src_name = random.choice(images)
                src_path = os.path.join(cls_path, src_name)
                
                # Create detailed name to track origin
                # Format: safe_aug_{batch}_{original_name}
                new_name = f"safe_aug_{generated}_{src_name}"
                dst_path = os.path.join(cls_path, new_name)
                
                if augment_image(src_path, dst_path):
                    generated += 1
                    pbar.update(1)
            
            pbar.close()
            print(f"   ✅ Augmented. New total: {count + generated}")
        else:
            print("   ✅ Count Sufficient.")

    print("\n" + "="*50)
    print("Balancing Complete")
    print("="*50)

if __name__ == "__main__":
    balance_dataset()
