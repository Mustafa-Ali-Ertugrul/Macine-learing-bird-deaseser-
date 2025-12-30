import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

def organize_dataset(csv_path, output_dir="organized_dataset"):
    """
    Organize images into class directories based on the CSV labels
    """
    # Validate inputs
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return None
    
    # Read the CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None
    
    # Check required columns
    if 'image_path' not in df.columns or 'disease' not in df.columns:
        print("❌ CSV must contain 'image_path' and 'disease' columns")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create class directories
    classes = df['disease'].unique()
    for class_name in classes:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
    
    # Copy images to class directories
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"Copying {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        image_path = row['image_path']
        disease = row['disease']
        
        # Check if image file exists
        if not os.path.exists(image_path):
            skipped_count += 1
            continue
        
        # Verify image can be opened
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            error_count += 1
            continue
        
        # Copy to class directory
        try:
            dest_path = output_path / disease / Path(image_path).name
            if not dest_path.exists():
                shutil.copy2(image_path, dest_path)
                copied_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            error_count += 1
    
    print(f"\n✅ Copied {copied_count} images to {output_dir}")
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} images (duplicates or missing)")
    if error_count > 0:
        print(f"❌ Encountered {error_count} errors")
    
    # Show class distribution
    print("\n📊 Class distribution:")
    for class_name in classes:
        class_dir = output_path / class_name
        image_count = len(list(class_dir.glob("*")))
        print(f"  {class_name}: {image_count} images")
    
    return output_dir

def create_train_val_test_splits(csv_path, output_dir="dataset_splits", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/validation/test splits
    """
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001:
        print("❌ Split ratios must sum to 1.0")
        return None
    
    # Read the CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None
    
    # Remove unknown images for splits
    labeled_df = df[df['disease'] != 'unknown'].copy()
    
    if len(labeled_df) == 0:
        print("❌ No labeled images found. Cannot create splits.")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Split into train (70%), validation (15%), test (15%)
    train_df, temp_df = train_test_split(
        labeled_df, 
        test_size=(1 - train_ratio), 
        random_state=42, 
        stratify=labeled_df['disease']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=test_ratio / (val_ratio + test_ratio), 
        random_state=42, 
        stratify=temp_df['disease']
    )
    
    # Save splits
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"✅ Created dataset splits:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Validation: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    # Show class distribution in train set
    print("\n📊 Train set class distribution:")
    print(train_df['disease'].value_counts())
    
    return output_path

def main():
    """
    Main function to organize the dataset
    """
    print("🐔 Poultry Disease Dataset Organizer")
    print("=" * 35)
    
    csv_path = "poultry_labeled_12k.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Show current statistics
    print(f"📊 Total images: {len(df)}")
    print(f"📊 Unknown images: {len(df[df['disease'] == 'unknown'])}")
    print(f"📊 Labeled images: {len(df[df['disease'] != 'unknown'])}")
    
    print("\n📋 Available actions:")
    print("1. Organize images into class directories")
    print("2. Create train/validation/test splits")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect an action (0-2): ").strip()
            
            if choice == "0":
                print("👋 Exiting...")
                break
            elif choice == "1":
                output_dir = input("Enter output directory name (default: organized_dataset): ").strip()
                output_dir = output_dir if output_dir else "organized_dataset"
                organize_dataset(csv_path, output_dir)
            elif choice == "2":
                output_dir = input("Enter output directory name (default: dataset_splits): ").strip()
                output_dir = output_dir if output_dir else "dataset_splits"
                create_train_val_test_splits(csv_path, output_dir)
            else:
                print("❌ Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()