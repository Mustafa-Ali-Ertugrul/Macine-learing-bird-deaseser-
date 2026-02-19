import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil

def download_public_chicken_dataset():
    """
    Download a small public chicken disease dataset
    """
    print("[Chicken] Downloading Public Chicken Disease Dataset")
    print("=" * 40)
    
    # Create directories
    dataset_dir = Path("chicken_disease_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Since we can't easily download from Kaggle without API keys,
    # let's create a simple dataset structure for demonstration
    print("📝 Creating a sample dataset structure...")
    
    # Create class directories
    classes = ["healthy", "salmonella", "ib", "ibd", "coccidiosis"]
    
    for class_name in classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    print("✅ Created sample dataset structure")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📂 Classes: {', '.join(classes)}")
    print("\n📝 Note:")
    print("This is a sample directory structure. To use real data:")
    print("1. Download actual poultry disease datasets from Kaggle")
    print("2. Extract them to the 'raw_dataset' folder")
    print("3. Run 'organize_poultry_dataset.py' to organize them")
    
    return dataset_dir

def main():
    """
    Main function
    """
    dataset_dir = download_public_chicken_dataset()
    
    print(f"\n🎉 Dataset ready at: {dataset_dir}")
    print("Next steps:")
    print("1. Add real images to the class folders")
    print("2. Use the HTML labeling tool for any additional labeling")
    print("3. Train your model with the organized dataset")

if __name__ == "__main__":
    main()