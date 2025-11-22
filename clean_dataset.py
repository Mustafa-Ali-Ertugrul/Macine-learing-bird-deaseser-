import os
from PIL import Image
from tqdm import tqdm

DATASET_DIR = 'Macine learing (bird deaseser)/final_poultry_dataset_10_classes'

def convert_to_jpg():
    print(f"🔍 Scanning {DATASET_DIR} and converting to RGB JPG...")
    
    converted_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in tqdm(files, desc="Processing"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
                path = os.path.join(root, file)
                
                try:
                    # Open and force convert to RGB
                    img = Image.open(path).convert('RGB')
                    
                    # If it's not already a JPG, or we just want to sanitize it
                    # We will overwrite/replace with a clean JPG
                    
                    # Construct new path
                    base_name = os.path.splitext(path)[0]
                    new_path = base_name + ".jpg"
                    
                    # Save as JPG
                    img.save(new_path, "JPEG", quality=95)
                    
                    # If original was not jpg (e.g. png), remove it
                    if path != new_path:
                        os.remove(path)
                        
                    converted_count += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {path}: {e}")
                    try:
                        os.remove(path)
                        print("   🗑️ Deleted corrupt file.")
                        error_count += 1
                    except:
                        pass

    print("\n" + "="*60)
    print(f"Conversion complete.")
    print(f"✅ Processed: {converted_count}")
    print(f"❌ Deleted: {error_count}")
    print("="*60)

if __name__ == "__main__":
    convert_to_jpg()
