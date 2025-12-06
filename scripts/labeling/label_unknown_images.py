import pandas as pd
import os
import random
from pathlib import Path

def sample_unknown_images(csv_path, sample_size=50):
    """
    Sample some unknown images for manual labeling
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for unknown images
    unknown_df = df[df['disease'] == 'unknown']
    
    # Sample some images for labeling
    if len(unknown_df) > sample_size:
        sampled_df = unknown_df.sample(n=sample_size, random_state=42)
    else:
        sampled_df = unknown_df
    
    # Save sampled images to a new CSV
    sampled_csv = "sampled_unknown_images.csv"
    sampled_df.to_csv(sampled_csv, index=False)
    
    print(f"✅ Sampled {len(sampled_df)} unknown images for labeling")
    print(f"📁 Sampled CSV: {sampled_csv}")
    
    return sampled_df

def create_labeling_template(csv_path):
    """
    Create a template CSV for labeling with disease categories
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for unknown images
    unknown_df = df[df['disease'] == 'unknown'].copy()
    
    # Add columns for labeling
    unknown_df['new_disease'] = ''  # To be filled by user
    unknown_df['notes'] = ''        # Optional notes
    
    # Save labeling template
    template_csv = "labeling_template.csv"
    unknown_df.to_csv(template_csv, index=False)
    
    print(f"✅ Created labeling template with {len(unknown_df)} images")
    print(f"📁 Template CSV: {template_csv}")
    print("\n📋 Disease categories to use:")
    print("  - healthy")
    print("  - ib (Infectious Bronchitis)")
    print("  - ibd (Infectious Bursal Disease)")
    print("  - coccidiosis")
    print("  - salmonella")
    print("  - fatty_liver")
    print("  - histomoniasis")
    print("  - other (for images that don't fit the above categories)")
    
    return unknown_df

def update_labeled_images(original_csv, labeled_csv, output_csv):
    """
    Update the original CSV with newly labeled images
    """
    # Read original CSV
    original_df = pd.read_csv(original_csv)
    
    # Read labeled CSV
    labeled_df = pd.read_csv(labeled_csv)
    
    # Update disease labels for matching image paths
    for idx, row in labeled_df.iterrows():
        image_path = row['image_path']
        new_disease = row['new_disease']
        
        # Update in original dataframe
        mask = original_df['image_path'] == image_path
        original_df.loc[mask, 'disease'] = new_disease
    
    # Save updated CSV
    original_df.to_csv(output_csv, index=False)
    print(f"✅ Updated {len(labeled_df)} image labels")
    print(f"📁 Updated CSV: {output_csv}")
    
    # Show new disease distribution
    print("\n📊 New disease distribution:")
    print(original_df['disease'].value_counts())
    
    return original_df

def main():
    """
    Main function to help with labeling unknown images
    """
    print("🐔 Poultry Disease Image Labeling Helper")
    print("=" * 40)
    
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
    print("1. Sample unknown images for labeling")
    print("2. Create labeling template")
    print("3. Update dataset with labeled images")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect an action (0-3): ").strip()
            
            if choice == "0":
                print("👋 Exiting...")
                break
            elif choice == "1":
                sample_size = input("Enter sample size (default 50): ").strip()
                sample_size = int(sample_size) if sample_size else 50
                sample_unknown_images(csv_path, sample_size)
            elif choice == "2":
                create_labeling_template(csv_path)
            elif choice == "3":
                labeled_csv = input("Enter path to labeled CSV: ").strip()
                if os.path.exists(labeled_csv):
                    output_csv = "updated_poultry_dataset.csv"
                    update_labeled_images(csv_path, labeled_csv, output_csv)
                else:
                    print(f"❌ File not found: {labeled_csv}")
            else:
                print("❌ Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()