import pandas as pd
import os
import random
from pathlib import Path

def create_labeling_batch(csv_path, batch_size=50):
    """
    Create a batch of images for labeling
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for unknown images
    unknown_df = df[df['disease'] == 'unknown']
    
    # Sample a batch for labeling
    if len(unknown_df) > batch_size:
        batch_df = unknown_df.sample(n=batch_size, random_state=42)
    else:
        batch_df = unknown_df
    
    # Save batch to a new CSV
    batch_csv = "labeling_batch.csv"
    batch_df.to_csv(batch_csv, index=False)
    
    print(f"✅ Created labeling batch with {len(batch_df)} images")
    print(f"📁 Batch CSV: {batch_csv}")
    
    return batch_df

def update_labels(original_csv, labeled_csv):
    """
    Update the original CSV with newly labeled images
    """
    # Read original CSV
    original_df = pd.read_csv(original_csv)
    
    # Read labeled CSV
    labeled_df = pd.read_csv(labeled_csv)
    
    # Update disease labels for matching image paths
    updated_count = 0
    for idx, row in labeled_df.iterrows():
        image_path = row['image_path']
        new_disease = row.get('new_disease', row.get('disease', 'unknown'))
        
        # Skip if still unknown
        if new_disease == 'unknown' or new_disease == '':
            continue
            
        # Update in original dataframe
        mask = original_df['image_path'] == image_path
        if mask.any():
            original_df.loc[mask, 'disease'] = new_disease
            updated_count += 1
    
    # Save updated CSV
    original_df.to_csv(original_csv, index=False)
    print(f"✅ Updated {updated_count} image labels in {original_csv}")
    
    # Show new disease distribution
    print("\n📊 New disease distribution:")
    print(original_df['disease'].value_counts())
    
    return original_df

def create_labeling_template(csv_path):
    """
    Create a template for labeling with instructions
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for unknown images
    unknown_df = df[df['disease'] == 'unknown'].copy()
    
    # Add columns for labeling
    unknown_df['new_disease'] = ''  # To be filled by user
    unknown_df['notes'] = ''        # Optional notes
    
    # Save labeling template
    template_csv = "poultry_labeling_template.csv"
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
    print("  - newcastle (Newcastle Disease)")
    print("  - marek (Marek's Disease)")
    print("  - avian_influenza (Avian Influenza)")
    print("  - other (for images that don't fit the above categories)")
    
    return unknown_df

def main():
    """
    Main function to start a labeling session
    """
    print("🐔 Poultry Disease Image Labeling Session")
    print("=" * 45)
    
    csv_path = "poultry_labeled_12k.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Show current statistics
    total_images = len(df)
    labeled_images = len(df[df['disease'] != 'unknown'])
    unlabeled_images = len(df[df['disease'] == 'unknown'])
    
    print(f"📊 Total images: {total_images}")
    print(f"📊 Labeled images: {labeled_images}")
    print(f"📊 Unlabeled images: {unlabeled_images}")
    
    print("\n📋 Available actions:")
    print("1. Create labeling template (CSV)")
    print("2. Create small labeling batch (50 images)")
    print("3. Update dataset with labeled images")
    print("4. Show disease distribution")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect an action (0-4): ").strip()
            
            if choice == "0":
                print("👋 Exiting...")
                break
            elif choice == "1":
                create_labeling_template(csv_path)
            elif choice == "2":
                batch_size = input("Enter batch size (default 50): ").strip()
                batch_size = int(batch_size) if batch_size else 50
                create_labeling_batch(csv_path, batch_size)
            elif choice == "3":
                labeled_csv = input("Enter path to labeled CSV: ").strip()
                if os.path.exists(labeled_csv):
                    update_labels(csv_path, labeled_csv)
                else:
                    print(f"❌ File not found: {labeled_csv}")
            elif choice == "4":
                print("\n📊 Disease distribution:")
                print(df['disease'].value_counts())
            else:
                print("❌ Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()