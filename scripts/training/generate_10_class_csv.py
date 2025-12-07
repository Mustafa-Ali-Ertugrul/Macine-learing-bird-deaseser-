import os
import pandas as pd
from pathlib import Path
from PIL import Image

def generate_csv():
    # Define paths
    # Note: The folder structure is nested: Macine learing (bird deaseser)/Macine learing (bird deaseser)/final_dataset_10_classes
    root_dir = Path(r"c:\Users\Ali\OneDrive\Belgeler\pyton\Macine learing (bird deaseser)\Macine learing (bird deaseser)\final_dataset_10_classes")
    output_csv = Path(r"c:\Users\Ali\OneDrive\Belgeler\pyton\Macine learing (bird deaseser)\data\metadata\final_dataset_10_classes.csv")
    
    data = []
    skipped_count = 0
    
    print(f"Scanning directory: {root_dir}")
    
    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist!")
        return

    # Iterate over disease folders
    for disease_dir in root_dir.iterdir():
        if disease_dir.is_dir():
            disease_name = disease_dir.name
            print(f"Found class: {disease_name}")
            
            # Iterate over images
            for image_file in disease_dir.glob("*.jpg"):
                
                # Validation 1: Size check (< 1KB is suspicious)
                if image_file.stat().st_size < 1024:
                    print(f"⚠️ Skipping too small file: {image_file.name} ({image_file.stat().st_size} bytes)")
                    skipped_count += 1
                    continue
                
                # Validation 2: PIL check
                try:
                    with Image.open(image_file) as img:
                        img.verify() # Verify integrity
                except Exception as e:
                    print(f"⚠️ Skipping corrupted file {image_file.name}: {e}")
                    skipped_count += 1
                    continue

                # Store relative path (Disease/Image.jpg)
                # The Dataset class joins root_dir + image_path
                relative_path = f"{disease_name}/{image_file.name}"
                
                data.append({
                    'image_path': relative_path,
                    'disease': disease_name,
                    # Add dummy values for other columns if needed, though train_model.py only uses image_path and disease
                    'tissue': 'unknown',
                    'source': 'final_10_classes',
                    'width': 512, # Assuming
                    'height': 512,
                    'filename': image_file.name,
                    'magnification': 'unknown'
                })
    
    print(f"\nSkipped {skipped_count} invalid images.")

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ CSV generated at: {output_csv}")
    print(f"Total images: {len(df)}")
    print(f"Classes found ({len(df['disease'].unique())}): {df['disease'].unique()}")

if __name__ == "__main__":
    generate_csv()
