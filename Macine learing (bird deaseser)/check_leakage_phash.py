import os
import sys
import torch
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DATA_DIR = 'final_dataset_clean_split'

def check_leakage_phash():
    print("="*60)
    print("🕵️ VISUAL LEAKAGE DETECTION (Perceptual Hash)")
    print("="*60)
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory '{DATA_DIR}' not found.")
        return

    # Check structure: train, val, test
    if not all(os.path.exists(os.path.join(DATA_DIR, s)) for s in ['train', 'val', 'test']):
        print("❌ Error: Expected 'train', 'val', 'test' subdirectories.")
        return

    # Get classes from train
    classes = [d for d in os.listdir(os.path.join(DATA_DIR, 'train')) if os.path.isdir(os.path.join(DATA_DIR, 'train', d))]
    
    total_leakage = 0
    
    for cls in classes:
        print(f"\nAnalyzing Class: {cls}")
        
        # Collect Train images
        train_path = os.path.join(DATA_DIR, 'train', cls)
        if not os.path.exists(train_path): continue
        train_imgs = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Collect Test/Val images
        test_imgs = []
        for split in ['val', 'test']:
            split_path = os.path.join(DATA_DIR, split, cls)
            if os.path.exists(split_path):
                test_imgs.extend([os.path.join(split_path, f) for f in os.listdir(split_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"   Train: {len(train_imgs)}, Test/Val: {len(test_imgs)}")
        
        # Calculate hashes for TRAIN
        train_hashes = {}
        for img_path in tqdm(train_imgs, desc="Hashing Train", leave=False):
            try:
                img = Image.open(img_path)
                h = imagehash.phash(img)
                train_hashes[h] = img_path
            except Exception as e:
                continue
                
        # Check TEST against TRAIN
        leakage_found = []
        for img_path in tqdm(test_imgs, desc="Checking Test", leave=False):
            try:
                img = Image.open(img_path)
                h = imagehash.phash(img)
                
                # Check for near-duplicates (Hamming distance < 5)
                # This is slow O(N*M), so we'll just check exact pHash and very close ones
                # Optimization: In a real scenario, use a VP-Tree or BK-Tree.
                # Here we just check exact pHash matches which resists resizing/minor edits
                
                if h in train_hashes:
                    leakage_found.append((img_path, train_hashes[h], 0)) # 0 distance
                    continue
                
                # Quick linear scan for very close matches (distance <= 2 is VERY similar)
                # Only check a subset if too slow? No, let's try strict first.
                # Actually, iterating all train hashes for every test image is O(N*M)
                # Let's just stick to hash collision for now which catches resizing.
                
            except:
                continue
        
        if leakage_found:
            print(f"⚠️  FOUND {len(leakage_found)} POTENTIAL LEAKS in {cls}")
            total_leakage += len(leakage_found)
            
            # Show first 3 leaks
            for i, (test_p, train_p, dist) in enumerate(leakage_found[:3]):
                print(f"   Leak {i+1}: Test '{os.path.basename(test_p)}' ~= Train '{os.path.basename(train_p)}'")
                
                # Visualize
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(Image.open(test_p))
                ax[0].set_title(f"TEST: {os.path.basename(test_p)}")
                ax[0].axis('off')
                
                ax[1].imshow(Image.open(train_p))
                ax[1].set_title(f"TRAIN: {os.path.basename(train_p)}")
                ax[1].axis('off')
                
                plt.suptitle(f"Potential Leak in {cls} (pHash Match)")
                plt.show() # This might not show in non-interactive, but we save it
                plt.savefig(f"leakage_example_{cls}_{i}.png")
                plt.close()

    print("\n" + "="*60)
    print(f"🏁 Total Potential Leaks Found: {total_leakage}")
    print("="*60)

if __name__ == "__main__":
    check_leakage_phash()
