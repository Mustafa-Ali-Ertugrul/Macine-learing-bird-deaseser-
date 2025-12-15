
import os
import shutil
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
QUARANTINE_DIR = 'Macine learing (bird deaseser)/quarantine_cleaned'

def setup_quarantine():
    if not os.path.exists(QUARANTINE_DIR):
        os.makedirs(QUARANTINE_DIR)
        print(f"Created quarantine directory: {QUARANTINE_DIR}")

def move_to_quarantine(file_path, reason):
    filename = os.path.basename(file_path)
    cls_name = os.path.basename(os.path.dirname(file_path))
    new_name = f"{cls_name}_{reason}_{filename}"
    dest = os.path.join(QUARANTINE_DIR, new_name)
    shutil.move(file_path, dest)
    return new_name


def clean_dataset():
    print(f"Starting generic dataset cleaning on: {DATA_DIR}")
    setup_quarantine()
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    total_removed = 0
    total_small = 0
    total_duplicates = 0
    
    for cls in classes:
        print(f"\nProcessing class: {cls}")
        cls_path = os.path.join(DATA_DIR, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        
        # Store hashes of KEPT images to check against
        kept_hashes = {} # hash_obj -> filename
        
        # Sort images to ensure deterministic behavior (e.g. keep '0.jpg' over 'safe_aug_...')
        # Prefer shorter filenames (likely originals) and then alphabetical
        images.sort(key=lambda x: (len(x), x))
        
        for img_name in tqdm(images):
            img_path = os.path.join(cls_path, img_name)
            
            try:
                # 1. Open and Validate
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # 2. Check Size
                    if width < 100 or height < 100:
                        move_to_quarantine(img_path, "too_small")
                        total_small += 1
                        total_removed += 1
                        continue
                        
                    # 3. Check Hash (Near-Deduplication)
                    current_hash = imagehash.phash(img)
                    
                    is_duplicate = False
                    collision_file = None
                    
                    # Check against all kept hashes in this class
                    # Optimization: This is O(N*M), but N is small (processed so far). 
                    # For 2500 images, ~3M comparisons max. fast enough for phash.
                    for kept_h, kept_name in kept_hashes.items():
                        dist = current_hash - kept_h
                        if dist <= 2: # Match threshold from report
                            is_duplicate = True
                            collision_file = kept_name
                            break
                    
                    if is_duplicate:
                        # Duplicate found!
                        # print(f"   Duplicate found: {img_name} ≈ {collision_file} (dist: {current_hash - list(kept_hashes.keys())[0] if len(kept_hashes)==1 else '?'})")
                        move_to_quarantine(img_path, "duplicate_near")
                        total_duplicates += 1
                        total_removed += 1
                        continue
                    
                    # If unique, register it
                    kept_hashes[current_hash] = img_name
                    
            except Exception as e:
                print(f"   Error processing {img_name}: {e}")
                # Don't quarantine immediately on error if it's just a read error? 
                # But if we can't read it, it's bad.
                # However, re-raising to catch file lock issues
                if "used by another process" in str(e):
                    print("   ⚠️ File locked, skipping.")
                else:
                    move_to_quarantine(img_path, "corrupt")
                    total_removed += 1

    print("\n" + "="*50)
    print("Cleaning Complete")
    print(f"Total Removed: {total_removed}")
    print(f"  - Small/Invalid: {total_small}")
    print(f"  - Duplicates: {total_duplicates}")
    print(f"Files moved to: {QUARANTINE_DIR}")
    print("="*50)


if __name__ == "__main__":
    clean_dataset()
