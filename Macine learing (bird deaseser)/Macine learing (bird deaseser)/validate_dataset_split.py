
import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import imagehash
from tqdm import tqdm
from PIL import Image

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_split'

class DatasetValidator:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
        
    def log_issue(self, severity, message):
        if severity == "ERROR":
            self.issues.append(f"🚨 {message}")
            print(f"🚨 {message}")
        else:
            self.warnings.append(f"⚠️  {message}")
            print(f"⚠️  {message}")
    
    def check_splits_exist(self):
        print("\n1️⃣  CHECKING SPLIT FOLDERS")
        for split in ['train', 'val', 'test']:
            path = os.path.join(DATA_DIR, split)
            if not os.path.exists(path):
                self.log_issue("ERROR", f"Missing split folder: {path}")
                return False
            print(f"✅ Found {split} set")
        return True

    def analyze_distribution_and_leakage(self):
        print("\n2️⃣  ANALYZING DISTRIBUTION & LEAKAGE")
        
        train_hashes = {}
        splits = ['train', 'val', 'test']
        
        total_counts = Counter()
        
        for split in splits:
            split_dir = os.path.join(DATA_DIR, split)
            classes = os.listdir(split_dir)
            
            print(f"\n   Analyzing {split} split ({len(classes)} classes)...")
            
            for cls in classes:
                cls_path = os.path.join(split_dir, cls)
                if not os.path.isdir(cls_path): continue
                
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
                count = len(images)
                total_counts[cls] += count
                
                # Check min counts
                if split == 'train' and count < 200:
                    self.log_issue("ERROR", f"Train class '{cls}' has only {count} images (Goal: 200)")
                
                # Leakage Check
                for img_name in images:
                    img_path = os.path.join(cls_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            h = str(imagehash.phash(img))
                            
                            if split == 'train':
                                train_hashes[h] = img_path
                            else:
                                if h in train_hashes:
                                    self.log_issue("ERROR", f"LEAKAGE: {img_name} ({split}) is in Train set!")
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")

        print("\n📊 Total Dataset Counts:")
        for cls, count in total_counts.items():
            print(f"   {cls}: {count}")

    def run(self):
        print("="*60)
        print("Dataset Validator (Physical Split Mode)")
        print("="*60)
        
        if not self.check_splits_exist():
            return
            
        self.analyze_distribution_and_leakage()
        
        print("\n" + "="*60)
        if self.issues:
            print(f"❌ Validation FAILED with {len(self.issues)} errors.")
        else:
            print("✅ Validation PASSED! No leakage, healthy counts.")

if __name__ == "__main__":
    DatasetValidator().run()
