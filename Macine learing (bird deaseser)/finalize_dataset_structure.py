import os
import shutil
from pathlib import Path
from tqdm import tqdm

SOURCE_DIR = 'final_dataset_10_classes'
TARGET_DIR = 'final_poultry_dataset_10_classes'

def finalize_dataset():
    print("📦 Finalizing dataset structure...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory {SOURCE_DIR} not found!")
        return

    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)
    
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    classes.sort()
    
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    total_images = 0
    
    for cls in classes:
        src_cls_path = os.path.join(SOURCE_DIR, cls)
        dst_cls_path = os.path.join(TARGET_DIR, cls)
        os.makedirs(dst_cls_path)
        
        files = [f for f in os.listdir(src_cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Processing {cls}: {len(files)} images")
        
        for i, filename in enumerate(tqdm(files, desc=cls, leave=False)):
            src_file = os.path.join(src_cls_path, filename)
            
            # Standardize filename: class_0001.jpg
            ext = os.path.splitext(filename)[1].lower()
            new_name = f"{cls.lower().replace(' ', '_')}_{i+1:04d}{ext}"
            dst_file = os.path.join(dst_cls_path, new_name)
            
            shutil.copy2(src_file, dst_file)
            total_images += 1
            
    print("\n" + "="*60)
    print("✅ DATASET FINALIZED")
    print("="*60)
    print(f"📁 Location: {os.path.abspath(TARGET_DIR)}")
    print(f"📊 Total Images: {total_images}")
    print(f"📚 Classes: {len(classes)}")
    print("="*60)

if __name__ == "__main__":
    finalize_dataset()
