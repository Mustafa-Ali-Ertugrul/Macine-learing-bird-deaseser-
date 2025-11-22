import os
import requests
import zipfile
from tqdm import tqdm
import pandas as pd

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
        print(f"[Error] Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_path):
    """
    Extract zip file
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"[Success] Extracted to {extract_path}")
        return True
    except Exception as e:
        print(f"[Error] Error extracting {zip_path}: {e}")
        return False

def create_dataset_csv(dataset_dir, csv_path):
    """
    Create a CSV file with image paths and classes
    """
    import pathlib
    
    rows = []
    # Look for class directories
    for class_dir in pathlib.Path(dataset_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            # Find all images in this class directory
            for img_path in class_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Get relative path from dataset_dir
                    rel_path = img_path.relative_to(dataset_dir)
                    rows.append({
                        'image_path': str(rel_path).replace('\\', '/'),
                        'class': class_name,
                        'filename': img_path.name
                    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[Success] Created dataset CSV with {len(df)} images: {csv_path}")
    return df

def main():
    """
    Main function to download a public poultry disease dataset
    """
    print("[Chicken] Downloading Public Poultry Disease Dataset")
    print("=" * 45)
    
    # Create directories
    dataset_dir = "public_poultry_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Try to download a public dataset
    # We'll use the Figshare dataset that we already have
    print("[Folder] Using existing Figshare dataset...")
    
    # Create CSV for the existing dataset
    csv_path = "poultry_labeled_12k.csv"
    
    # Check if we have the poultry_labeled.csv file
    if os.path.exists("poultry_labeled.csv"):
        print("[Stats] Creating dataset CSV from existing data...")
        df = pd.read_csv("poultry_labeled.csv")
        
        # Add source column
        df['source'] = 'figshare'
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"[Success] Updated dataset CSV: {csv_path}")
        print(f"[Stats] Total images: {len(df)}")
        
        # Show disease distribution
        if 'disease' in df.columns:
            print("\n[Stats] Disease distribution:")
            print(df['disease'].value_counts())
    else:
        print("[Error] Could not find poultry_labeled.csv")
        print("Please run the poultry_bulk_downloader.py script first")
    
    print("\n[Complete] Dataset preparation complete!")
    print(f"[Folder] Dataset CSV: {csv_path}")

if __name__ == "__main__":
    main()