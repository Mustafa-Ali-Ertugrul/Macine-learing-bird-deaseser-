import os
import sys
from pathlib import Path
import subprocess
import json
import zipfile
from tqdm import tqdm
import shutil

def setup_kaggle_api():
    """
    Check if Kaggle API is properly configured
    """
    kaggle_config_path = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_config_path / "kaggle.json"
    
    if not kaggle_json_path.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nTo use the Kaggle API, you need to:")
        print("1. Go to https://www.kaggle.com/account and sign in")
        print("2. Click on 'Create New API Token' to download kaggle.json")
        print(f"3. Place the kaggle.json file in: {kaggle_config_path}")
        print("\nAfter setting up the credentials, run this script again.")
        return False
    return True

def search_poultry_datasets():
    """
    Search for poultry-related datasets on Kaggle
    """
    try:
        # Search for poultry disease datasets
        result = subprocess.run([
            "kaggle", "datasets", "list", 
            "-s", "poultry disease", 
            "--json"
        ], capture_output=True, text=True, check=True)
        
        datasets = json.loads(result.stdout)
        return datasets
    except subprocess.CalledProcessError as e:
        print(f"Error searching datasets: {e}")
        return []
    except json.JSONDecodeError:
        # Fallback search
        try:
            result = subprocess.run([
                "kaggle", "datasets", "list", 
                "-s", "chicken disease", 
                "--json"
            ], capture_output=True, text=True, check=True)
            
            datasets = json.loads(result.stdout)
            return datasets
        except Exception as e:
            print(f"Fallback search also failed: {e}")
            return []

def download_dataset(dataset_ref, download_path):
    """
    Download a dataset from Kaggle
    """
    try:
        print(f"⬇️  Downloading dataset: {dataset_ref}")
        
        # Create download directory
        download_path = Path(download_path)
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        result = subprocess.run([
            "kaggle", "datasets", "download", 
            dataset_ref,
            "-p", str(download_path)
        ], capture_output=True, text=True, check=True)
        
        print("✅ Download completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False

def extract_dataset(zip_path, extract_path):
    """
    Extract dataset files from zip archive
    """
    try:
        print(f"📦 Extracting: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            
            # Create progress bar
            with tqdm(total=len(file_list), desc="Extracting files") as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_path)
                    pbar.update(1)
        
        print("✅ Extraction completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False

def find_image_files(directory):
    """
    Find all image files in a directory
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files

def organize_poultry_images(source_dir, target_dir):
    """
    Organize poultry images into a structured directory
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = find_image_files(source_dir)
    
    print(f"📂 Found {len(image_files)} image files")
    
    # Copy images to target directory
    for i, image_path in enumerate(tqdm(image_files, desc="Organizing images")):
        src = Path(image_path)
        dst = target_dir / f"poultry_image_{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    
    print(f"✅ Organized {len(image_files)} images to {target_dir}")

def main():
    """
    Main function to download and organize poultry disease datasets
    """
    print("🐔 Poultry Disease Dataset Downloader")
    print("=" * 40)
    
    # Check Kaggle API setup
    if not setup_kaggle_api():
        return
    
    # Search for datasets
    print("🔍 Searching for poultry disease datasets...")
    datasets = search_poultry_datasets()
    
    if not datasets:
        print("❌ No datasets found. Trying alternative search terms...")
        # Try alternative search terms
        search_terms = ["chicken disease", "poultry pathology", "bird disease"]
        all_datasets = []
        
        for term in search_terms:
            print(f"🔍 Searching for '{term}'...")
            try:
                result = subprocess.run([
                    "kaggle", "datasets", "list", 
                    "-s", term, 
                    "--json"
                ], capture_output=True, text=True, check=True)
                
                datasets = json.loads(result.stdout)
                all_datasets.extend(datasets)
            except:
                continue
        
        datasets = all_datasets
    
    if not datasets:
        print("❌ No datasets found with any search terms")
        return
    
    print(f"✅ Found {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets[:5]):  # Show top 5
        ref = dataset.get('ref', 'Unknown')
        title = dataset.get('title', 'Untitled')
        print(f"  {i+1}. {title} ({ref})")
    
    # For now, let's download a known good dataset
    # You can modify this to download specific datasets
    datasets_to_download = [
        "arjunpandit/chicken-disease-dataset",
        # Add more datasets here as needed
    ]
    
    downloads_dir = Path("kaggle_downloads")
    organized_dir = Path("poultry_disease_images")
    
    # Download datasets
    for dataset_ref in datasets_to_download:
        print(f"\n📥 Processing dataset: {dataset_ref}")
        
        # Download
        if download_dataset(dataset_ref, downloads_dir):
            # Find the zip file
            zip_files = list(downloads_dir.glob("*.zip"))
            
            if zip_files:
                zip_path = zip_files[0]
                extract_dir = downloads_dir / "extracted"
                
                # Extract
                if extract_dataset(zip_path, extract_dir):
                    # Organize images
                    organize_poultry_images(extract_dir, organized_dir)
                
                # Clean up zip file
                zip_path.unlink()
            else:
                print("❌ No zip file found after download")
    
    print(f"\n🎉 All done!")
    print(f"📦 Images organized in: {organized_dir}")
    print(f"📊 Next steps:")
    print(f"   1. Review the images in {organized_dir}")
    print(f"   2. Use the HTML labeling tool to annotate them")
    print(f"   3. Train your poultry disease classification model")

if __name__ == "__main__":
    main()