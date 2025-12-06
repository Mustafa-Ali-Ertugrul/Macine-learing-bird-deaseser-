import pandas as pd
import os
from pathlib import Path

def label_images_interactively():
    """
    Interactive tool to label poultry disease images
    """
    csv_path = "poultry_labeled_12k.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for unknown images
    unknown_df = df[df['disease'] == 'unknown']
    
    if len(unknown_df) == 0:
        print("✅ All images have been labeled!")
        return
    
    print("🐔 Interactive Poultry Disease Image Labeling Tool")
    print("=" * 50)
    print(f"Found {len(unknown_df)} unlabeled images")
    print("\nDisease categories:")
    print("  1. healthy")
    print("  2. ib (Infectious Bronchitis)")
    print("  3. ibd (Infectious Bursal Disease)")
    print("  4. coccidiosis")
    print("  5. salmonella")
    print("  6. fatty_liver")
    print("  7. histomoniasis")
    print("  8. skip (for now)")
    print("  0. exit")
    
    # Disease mapping
    disease_map = {
        '1': 'healthy',
        '2': 'ib',
        '3': 'ibd',
        '4': 'coccidiosis',
        '5': 'salmonella',
        '6': 'fatty_liver',
        '7': 'histomoniasis'
    }
    
    labeled_count = 0
    processed_count = 0
    
    # Process images one by one
    for idx, row in unknown_df.iterrows():
        image_path = row['image_path']
        filename = row['filename']
        
        print(f"\n--- Image {processed_count + 1}/{len(unknown_df)} ---")
        print(f"Filename: {filename}")
        print(f"Path: {image_path}")
        
        # Show image info if available
        if os.path.exists(image_path):
            try:
                # Get image size
                from PIL import Image
                img = Image.open(image_path)
                print(f"Image size: {img.size[0]}x{img.size[1]}")
                img.close()
            except:
                print("Image size: Unknown")
        else:
            print("⚠️  Image file not found!")
        
        # Get user input
        while True:
            choice = input("\nSelect disease category (0-8): ").strip()
            
            if choice == "0":
                print("👋 Exiting...")
                # Save progress
                df.to_csv(csv_path, index=False)
                print(f"💾 Saved progress: {labeled_count} images labeled")
                return
            elif choice == "8":
                print("⏭️  Skipping this image...")
                processed_count += 1
                break
            elif choice in disease_map:
                disease = disease_map[choice]
                # Update in dataframe
                df.loc[df['image_path'] == image_path, 'disease'] = disease
                print(f"✅ Labeled as: {disease}")
                labeled_count += 1
                processed_count += 1
                break
            else:
                print("❌ Invalid choice. Please select 0-8.")
    
    # Save final results
    df.to_csv(csv_path, index=False)
    print(f"\n🎉 Labeling session completed!")
    print(f"📊 Total images labeled: {labeled_count}")
    print(f"💾 Dataset saved to: {csv_path}")
    
    # Show new disease distribution
    print(f"\n📊 New disease distribution:")
    disease_counts = df['disease'].value_counts()
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count}")

def main():
    """
    Main function
    """
    try:
        label_images_interactively()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Progress saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()