import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil
from urllib.parse import urlparse

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
    Main function to download public poultry disease dataset
    """
    print("🐔 Simple Poultry Disease Dataset Downloader")
    print("=" * 45)
    
    # Create directories
    downloads_dir = Path("public_downloads")
    downloads_dir.mkdir(exist_ok=True)
    
    # Public dataset URL (Chicken Disease Classification)
    # This is a popular dataset on Kaggle but we'll download it directly
    dataset_url = "https://github.com/PrathamDogra/Poultry-Disease-Classification-Dataset/archive/refs/heads/main.zip"
    zip_filename = downloads_dir / "poultry_disease_dataset.zip"
    
    print(f"📥 Downloading poultry disease dataset...")
    print(f"🔗 URL: {dataset_url}")
    
    # Download dataset
    if download_file(dataset_url, zip_filename):
        print("✅ Download completed!")
        
        # Extract dataset
        extract_dir = downloads_dir / "extracted"
        if extract_zip(zip_filename, extract_dir):
            # Find the actual dataset directory
            dataset_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            
            if dataset_dirs:
                dataset_dir = dataset_dirs[0]
                print(f"📂 Dataset directory: {dataset_dir}")
                
                # Organize images
                organized_dir = Path("organized_poultry_images")
                organize_images(dataset_dir, organized_dir)
                
                # Clean up
                shutil.rmtree(downloads_dir)
                
                print(f"\n🎉 Success!")
                print(f"📁 Organized images: {organized_dir}")
                print(f"📊 Image count: {len(list(organized_dir.glob('*')))}")
                print(f"\nNext steps:")
                print(f"1. Review the images in {organized_dir}")
                print(f"2. Use the HTML labeling tool to annotate them")
                print(f"3. Train your poultry disease classification model")
            else:
                print("❌ No dataset directory found after extraction")
    else:
        print("❌ Failed to download dataset")
        print("Trying alternative source...")
        
        # Alternative dataset
        alt_url = "https://github.com/PrathamDogra/Poultry-Disease-Classification-Dataset/archive/main.zip"
        print(f"🔗 Trying alternative URL: {alt_url}")
        
        if download_file(alt_url, zip_filename):
            print("✅ Download completed!")
            
            # Extract dataset
            extract_dir = downloads_dir / "extracted"
            if extract_zip(zip_filename, extract_dir):
                # Find the actual dataset directory
                dataset_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
                
                if dataset_dirs:
                    dataset_dir = dataset_dirs[0]
                    print(f"📂 Dataset directory: {dataset_dir}")
                    
                    # Organize images
                    organized_dir = Path("organized_poultry_images")
                    organize_images(dataset_dir, organized_dir)
                    
                    # Clean up
                    shutil.rmtree(downloads_dir)
                    
                    print(f"\n🎉 Success!")
                    print(f"📁 Organized images: {organized_dir}")
                    print(f"📊 Image count: {len(list(organized_dir.glob('*')))}")
                    print(f"\nNext steps:")
                    print(f"1. Review the images in {organized_dir}")
                    print(f"2. Use the HTML labeling tool to annotate them")
                    print(f"3. Train your poultry disease classification model")

if __name__ == "__main__":
    main()