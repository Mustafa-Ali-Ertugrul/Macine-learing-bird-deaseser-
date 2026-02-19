import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_dataset(csv_path, output_dir="organized_dataset"):
    """
    Organize images into class directories based on the CSV labels
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create class directories
    classes = df['disease'].unique()
    for class_name in classes:
        class_dir = Path(output_dir) / class_name
        class_dir.mkdir(exist_ok=True)
    
    # Copy images to class directories
    copied_count = 0
    for idx, row in df.iterrows():
        image_path = row['image_path']
        disease = row['disease']
        
        # Check if image file exists
        if os.path.exists(image_path):
            # Copy to class directory
            dest_path = Path(output_dir) / disease / Path(image_path).name
            shutil.copy2(image_path, dest_path)
            copied_count += 1
    
    print(f"✅ Copied {copied_count} images to {output_dir}")
    
    # Show class distribution
    print("\n📊 Class distribution:")
    for class_name in classes:
        class_dir = Path(output_dir) / class_name
        image_count = len(list(class_dir.glob("*")))
        print(f"  {class_name}: {image_count} images")
    
    return output_dir

def create_train_val_test_splits(csv_path, output_dir="dataset_splits"):
    """
    Create train/validation/test splits
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Remove unknown images for splits
    labeled_df = df[df['disease'] != 'unknown']
    
    if len(labeled_df) == 0:
        print("❌ No labeled images found. Cannot create splits.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Split into train (70%), validation (15%), test (15%)
    train_df, temp_df = train_test_split(
        labeled_df, 
        test_size=0.3, 
        random_state=42, 
        stratify=labeled_df['disease']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['disease']
    )
    
    # Save splits
    train_df.to_csv(Path(output_dir) / "train.csv", index=False)
    val_df.to_csv(Path(output_dir) / "val.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test.csv", index=False)
    
    print(f"✅ Created dataset splits:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Validation: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    # Show class distribution in train set
    print("\n📊 Train set class distribution:")
    print(train_df['disease'].value_counts())
    
    return output_dir

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