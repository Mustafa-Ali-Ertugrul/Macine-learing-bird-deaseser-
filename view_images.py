import pandas as pd
import os
from PIL import Image

def view_images_for_labeling(csv_path, num_images=5):
    """
    Display information about images for labeling
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for unknown images
    unknown_df = df[df['disease'] == 'unknown']
    
    # Sample some images
    if len(unknown_df) > num_images:
        sample_df = unknown_df.sample(n=num_images)
    else:
        sample_df = unknown_df
    
    print(f"Showing {len(sample_df)} images for labeling:")
    print("Disease categories: healthy, ib, ibd, coccidiosis, salmonella, fatty_liver, histomoniasis, newcastle, marek, avian_influenza")
    print("\nImage details:")
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        image_path = row['image_path']
        print(f"{i+1}. Filename: {row['filename']}")
        print(f"   Path: {image_path}")
        print(f"   Suggested action: Open this image in an image viewer to examine it")
        print()
    
    return sample_df

def main():
    """
    Main function to view images for labeling
    """
    print("🐔 View Poultry Disease Images for Labeling")
    print("=" * 45)
    
    csv_path = "poultry_labeled_12k.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Show statistics
    total_images = len(df)
    labeled_images = len(df[df['disease'] != 'unknown'])
    unlabeled_images = len(df[df['disease'] == 'unknown'])
    
    print(f"📊 Total images: {total_images}")
    print(f"📊 Labeled images: {labeled_images}")
    print(f"📊 Unlabeled images: {unlabeled_images}")
    
    # View some images
    try:
        view_images_for_labeling(csv_path, 5)
    except Exception as e:
        print(f"❌ Error displaying images: {e}")

if __name__ == "__main__":
    main()