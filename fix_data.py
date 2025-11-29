import os
import shutil
from PIL import Image, ImageEnhance
import random
import glob

DATA_DIR = r"c:\Users\Ali\OneDrive\Belgeler\pyton\Macine learing (bird deaseser)\Macine learing (bird deaseser)\final_poultry_dataset_10_classes\Avian Influenza"
TARGET_COUNT = 500

def cleanup_and_augment():
    print(f"Scanning {DATA_DIR}...")
    
    # 1. Find all images recursively
    all_images = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                try:
                    img = Image.open(full_path).convert('RGB')
                    all_images.append(img)
                except Exception as e:
                    print(f"Skipping bad file {full_path}: {e}")

    print(f"Found {len(all_images)} valid images.")

    if len(all_images) == 0:
        print("❌ No images found! Please check the folder.")
        return

    # 2. Clear directory (remove subfolders)
    print("Cleaning up directory...")
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")

    # 3. Save original images
    print("Saving original images...")
    for i, img in enumerate(all_images):
        save_path = os.path.join(DATA_DIR, f"avian_influenza_new_{i+1:04d}.jpg")
        img.save(save_path)

    current_count = len(all_images)
    
    # 4. Augment if needed
    if current_count < TARGET_COUNT:
        needed = TARGET_COUNT - current_count
        print(f"Need {needed} more images to reach {TARGET_COUNT}. Augmenting...")
        
        generated = 0
        while generated < needed:
            # Pick a random original image
            src_img = random.choice(all_images)
            
            # Apply random augmentation
            aug_type = random.choice(['rotate', 'flip', 'brightness', 'contrast', 'sharpness'])
            
            if aug_type == 'rotate':
                angle = random.randint(-30, 30)
                new_img = src_img.rotate(angle)
            elif aug_type == 'flip':
                new_img = src_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif aug_type == 'brightness':
                factor = random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Brightness(src_img)
                new_img = enhancer.enhance(factor)
            elif aug_type == 'contrast':
                factor = random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Contrast(src_img)
                new_img = enhancer.enhance(factor)
            elif aug_type == 'sharpness':
                factor = random.uniform(0.5, 2.0)
                enhancer = ImageEnhance.Sharpness(src_img)
                new_img = enhancer.enhance(factor)
            
            # Save
            save_path = os.path.join(DATA_DIR, f"avian_influenza_aug_{generated+1:04d}.jpg")
            new_img.save(save_path)
            generated += 1
            
        print(f"✅ Successfully generated {generated} images.")
    
    final_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')])
    print(f"Final count in {DATA_DIR}: {final_count}")

if __name__ == "__main__":
    cleanup_and_augment()
