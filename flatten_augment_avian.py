
import os
import shutil
from PIL import Image, ImageEnhance
import random

TARGET_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
AVIAN_DIR_NAME = 'Avian Influenza in Poultry _ The Poultry Site'
MIN_IMAGES = 500

def augment_image(image_path, save_path):
    try:
        img = Image.open(image_path).convert('RGB')
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
    avian_path = os.path.join(TARGET_DIR, AVIAN_DIR_NAME)
    if not os.path.exists(avian_path):
        print(f"❌ Path not found: {avian_path}")
        return

    # 1. Flatten Subdirectories
    print(f"📂 Flattening subdirectories in {avian_path}...")
    moved_count = 0
    for root, dirs, files in os.walk(avian_path):
        if root == avian_path:
            continue # Don't move files already in root
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                # Handle unique filename
                base_name = file
                dst_path = os.path.join(avian_path, base_name)
                counter = 1
                while os.path.exists(dst_path):
                    name, ext = os.path.splitext(base_name)
                    dst_path = os.path.join(avian_path, f"{name}_{counter}{ext}")
                    counter += 1
                
                try:
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {src_path}: {e}")
    
    # Remove empty subdirs
    for root, dirs, files in os.walk(avian_path, topdown=False):
        if root == avian_path: continue
        try:
            os.rmdir(root)
        except:
            pass
            
    print(f"✅ Moved {moved_count} images to root of class folder.")

    # 2. Check and Augment
    images = [f for f in os.listdir(avian_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = len(images)
    print(f"Current Avian Influenza Count: {count}")

    if count < MIN_IMAGES:
        needed = MIN_IMAGES - count
        print(f"⚠️ Augmenting {needed} images...")
        if count == 0:
             print("❌ Still 0 images! Cannot augment.")
             return

        generated = 0
        while generated < needed:
            src_name = random.choice(images)
            src_full = os.path.join(avian_path, src_name)
            new_name = f"aug_flat_{generated}_{src_name}"
            dst_full = os.path.join(avian_path, new_name)
            
            if augment_image(src_full, dst_full):
                generated += 1
        
        print(f"✅ Augmented to reach {count + generated} images.")
    else:
        print("✅ Count is sufficient.")

if __name__ == "__main__":
    main()
