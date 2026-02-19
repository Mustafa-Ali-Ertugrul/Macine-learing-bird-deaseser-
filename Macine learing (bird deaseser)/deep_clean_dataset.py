
import os
import hashlib
from PIL import Image

from collections import defaultdict
import shutil

DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
QUARANTINE_DIR = 'Macine learing (bird deaseser)/quarantine_duplicates'

def get_file_hash(path):
    """Calculate MD5 hash of file content"""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def deep_clean():
    print(f"🕵️ Starting Deep Forensic Analysis on {DATA_DIR}...")
    
    if not os.path.exists(QUARANTINE_DIR):
        os.makedirs(QUARANTINE_DIR)
        
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # 1. Exact Duplicate Detection (MD5)
    print("\n🔍 Phase 1: Checking for OPTICAL DUPLICATES (MD5 Hash)...")
    hash_map = defaultdict(list)
    total_files = 0
    
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_files += len(files)
        
        for f in files:
            path = os.path.join(cls_path, f)
            file_hash = get_file_hash(path)
            hash_map[file_hash].append(path)
            
    # Find duplicates
    duplicates_found = 0
    for file_hash, paths in hash_map.items():
        if len(paths) > 1:
            # Keep the first one, move others to quarantine
            original = paths[0]
            copies = paths[1:]
            
            print(f"   ⚠️ Found {len(copies)} copies of {os.path.basename(original)}")
            for copy_path in copies:
                # Move to quarantine
                filename = os.path.basename(copy_path)
                cls_name = os.path.basename(os.path.dirname(copy_path))
                dest = os.path.join(QUARANTINE_DIR, f"{cls_name}_{filename}")
                shutil.move(copy_path, dest)
                duplicates_found += 1
                
    print(f"✅ Phase 1 Complete. Removed {duplicates_found} exact duplicates.")
    
    # 2. Re-verify Counts
    print("\n📊 Final Status:")
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        count = len(os.listdir(cls_path))
        print(f"   {cls:<30}: {count} images")
        
    print("-" * 50)
    print(f"🗑️ Duplicates moved to: {QUARANTINE_DIR}")
    print("🚀 You can now restart training with a truly clean dataset.")

if __name__ == "__main__":
    deep_clean()
