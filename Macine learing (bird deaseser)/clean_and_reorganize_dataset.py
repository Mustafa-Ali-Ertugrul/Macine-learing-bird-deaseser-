import os
import shutil
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
from tqdm import tqdm

# Configuration
SOURCE_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
TARGET_DIR = 'final_dataset_clean_split'
AUGMENT_FACTOR = 5  # Number of augmentations per image for training only

print("=" * 60)
print("🐔 DATASET CLEANING & RESTRUCTURING PIPELINE")
print("=" * 60)

# 1. Clean and Collect Originals
print("\n🔍 Step 1: Collecting original images (ignoring 'safe_aug_')...")
data_map = {} # class -> list of file paths

if not os.path.exists(SOURCE_DIR):
    print(f"❌ Error: Source directory '{SOURCE_DIR}' not found.")
    exit(1)

total_files = 0
kept_files = 0

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
print(f"   Found classes: {classes}")

for cls in classes:
    src_path = os.path.join(SOURCE_DIR, cls)
    data_map[cls] = []
    
    for f in os.listdir(src_path):
        total_files += 1
        if f.lower().startswith('safe_aug_') or 'aug' in f.lower() and ('sample' in f.lower() or 'copy' in f.lower()):
            continue # Skip augmented files
            
        full_path = os.path.join(src_path, f)
        data_map[cls].append(full_path)
        kept_files += 1

print(f"   Total files scanned: {total_files}")
print(f"   Originals kept: {kept_files}")
print(f"   Augmented files ignored: {total_files - kept_files}")

# 2. Split Data (Stratified)
print("\n✂️  Step 2: Splitting Data (70% Train, 15% Val, 15% Test)...")

splits = {'train': {}, 'val': {}, 'test': {}}

for cls, files in data_map.items():
    if len(files) < 10:
        print(f"⚠️  Warning: Class {cls} has only {len(files)} images. Performing minimal non-overlapping split to prevent leakage.")
        
        # Shuffle to ensure randomness
        random.shuffle(files)
        
        # Strategy: Ensure NO overlap.
        # Priority: Train > Test > Val
        
        if len(files) >= 3:
            # At least 1 for each
            splits['test'][cls] = [files[0]]
            splits['val'][cls] = [files[1]]
            splits['train'][cls] = files[2:]
        elif len(files) == 2:
            # 1 Train, 1 Test, 0 Val
            splits['test'][cls] = [files[0]]
            splits['val'][cls] = []
            splits['train'][cls] = [files[1]]
        elif len(files) == 1:
             # 1 Train, 0 Test, 0 Val (Training is most critical)
            splits['test'][cls] = []
            splits['val'][cls] = []
            splits['train'][cls] = files
        else:
            # Should not happen as we check for empty folders earlier, but for safety
            splits['train'][cls] = [] 
            splits['val'][cls] = []
            splits['test'][cls] = []

        print(f"   {cls:<25}: {len(splits['train'][cls])} Train, {len(splits['val'][cls])} Val, {len(splits['test'][cls])} Test")
        continue

    # Split: Train (70%) + Temp (30%)
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42, shuffle=True)
    # Split Temp: Val (50% of Temp = 15% total) + Test (50% of Temp = 15% total)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42, shuffle=True)
    
    splits['train'][cls] = train_files
    splits['val'][cls] = val_files
    splits['test'][cls] = test_files
    
    print(f"   {cls:<25}: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

# 3. Create Directories and Copy
print("\n📂 Step 3: Creating target structure and copying files...")

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
    print(f"   Removed existing '{TARGET_DIR}'")

for split_name, class_data in splits.items():
    for cls, files in class_data.items():
        dst_path = os.path.join(TARGET_DIR, split_name, cls)
        os.makedirs(dst_path, exist_ok=True)
        
        for f in files:
            shutil.copy2(f, dst_path)

print("   Copy complete.")

# 4. Augment ONLY Training Data
print("\n🎨 Step 4: Augmenting TRAINING set only...")

augment_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
])

for cls in classes:
    train_cls_dir = os.path.join(TARGET_DIR, 'train', cls)
    files = [f for f in os.listdir(train_cls_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"   Augmenting {cls} (Originals: {len(files)})...")
    
    count = 0
    target_count = max(500, len(files) * 3) # Ensure at least 500 images per class or 3x original
    
    # Calculate how many augs per image needed
    needed = target_count - len(files)
    if needed <= 0:
        continue
        
    augs_per_img = max(1, needed // len(files)) + 1
    
    for fname in tqdm(files, desc=f"   {cls}", leave=False):
        img_path = os.path.join(train_cls_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            
            for i in range(augs_per_img):
                aug_img = augment_pipeline(img)
                save_name = f"safe_aug_{i}_{fname}"
                aug_img.save(os.path.join(train_cls_dir, save_name))
                count += 1
                if len(os.listdir(train_cls_dir)) >= target_count:
                    break
        except Exception as e:
            print(f"   Error augmenting {fname}: {e}")
            
print("\n✅ Dataset reconstruction complete!")
print(f"   Location: {os.path.abspath(TARGET_DIR)}")
print("   Leakage check: Pre-augmentation files are kept strict. Only Train is augmented.")
