import os
import glob
from tqdm import tqdm

DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'

# Keywords that indicate an image is NOT original
BAD_KEYWORDS = ['aug', 'rot', 'flip', 'zoom', 'augmented', 'copy', 'horizontal', 'vertical', 'balanced']

def clean_augmented_files():
    print(f"🧹 STARTING AGGRESSIVE CLEANUP on {DATA_DIR}")
    print(f"keywords: {BAD_KEYWORDS}")
    
    deleted_count = 0
    total_space_saved = 0
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        print(f"\nScanning {cls}...")
        
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        initial_count = len(images)
        cls_deleted = 0
        
        for img in images:
            # Check if filename contains any bad keyword
            if any(kw in img.lower() for kw in BAD_KEYWORDS):
                file_path = os.path.join(cls_path, img)
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    cls_deleted += 1
                    total_space_saved += size
                except Exception as e:
                    print(f"Error removing {img}: {e}")
        
        print(f"   Removed {cls_deleted} / {initial_count} images.")

    print("\n" + "="*50)
    print(f"🏁 DONE.")
    print(f"🗑️  Total Deleted: {deleted_count} files")
    print(f"💾 Space Reclaimed: {total_space_saved / (1024*1024):.2f} MB")
    print("="*50)

if __name__ == '__main__':
    clean_augmented_files()
