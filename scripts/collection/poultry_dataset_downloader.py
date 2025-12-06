import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil

def download_file(url, filename):
    """
    Download a file from URL with progress bar
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Save file with progress bar
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_path):
    """
    Extract zip file
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ Extracted to {extract_path}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False

def find_images(directory):
    """
    Find all image files in directory
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                images.append(os.path.join(root, file))
    
    return images

def organize_images(source_dir, target_dir):
    """
    Organize images into a single directory
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    images = find_images(source_dir)
    
    print(f"📂 Found {len(images)} images")
    
    for i, image_path in enumerate(tqdm(images, desc="Organizing images")):
        src = Path(image_path)
        dst = target_dir / f"poultry_disease_{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    
    print(f"✅ Organized images to {target_dir}")

def main():
    """
    Main function to download a public poultry disease dataset
    """
    print("🐔 Public Poultry Disease Dataset Downloader")
    print("=" * 45)
    
    # Create directories
    downloads_dir = Path("dataset_downloads")
    downloads_dir.mkdir(exist_ok=True)
    
    # We'll use a different approach - let's create a script that helps you
    # download datasets manually and organize them
    print("📝 Instructions for downloading poultry disease datasets:")
    print("")
    print("1. Visit these websites to download poultry disease datasets:")
    print("   - https://www.kaggle.com/datasets/arjunpandit/chicken-disease-dataset")
    print("   - https://www.kaggle.com/datasets/dangvh2000/poultry-disease-classification")
    print("   - https://www.kaggle.com/datasets/gpiosenka/chicken-disease-image-dataset")
    print("")
    print("2. Download any of these datasets (you may need to sign in to Kaggle)")
    print("3. Extract the downloaded ZIP file to a folder")
    print("4. Run this script again after placing the extracted dataset in a folder named 'raw_dataset'")
    print("")
    
    # Check if user has already placed a dataset in raw_dataset folder
    raw_dataset_dir = Path("raw_dataset")
    if raw_dataset_dir.exists() and any(raw_dataset_dir.iterdir()):
        print("🔍 Found raw dataset folder. Organizing images...")
        
        # Organize images
        organized_dir = Path("organized_poultry_images")
        organize_images(raw_dataset_dir, organized_dir)
        
        print(f"\n🎉 Success!")
        print(f"📁 Organized images: {organized_dir}")
        print(f"📊 Image count: {len(list(organized_dir.glob('*')))}")
        print(f"\nNext steps:")
        print(f"1. Review the images in {organized_dir}")
        print(f"2. Use the HTML labeling tool to annotate them")
        print(f"3. Train your poultry disease classification model")
    else:
        print("📁 To proceed, please:")
        print(f"   1. Create a folder named 'raw_dataset' in this directory")
        print(f"   2. Place your downloaded poultry disease dataset in this folder")
        print(f"   3. Run this script again")

if __name__ == "__main__":
    main()