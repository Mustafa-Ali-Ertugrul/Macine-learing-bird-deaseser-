import pandas as pd
import os

def update_dataset_labels():
    """
    Update the main dataset with newly labeled images
    """
    # Check if labeling template exists
    template_path = "poultry_labeling_template.csv"
    dataset_path = "poultry_labeled_12k.csv"
    
    if not os.path.exists(template_path):
        print(f"❌ Labeling template not found: {template_path}")
        print("Please create a labeling template first.")
        return
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Read both CSV files
    template_df = pd.read_csv(template_path)
    dataset_df = pd.read_csv(dataset_path)
    
    # Count how many images have been labeled
    labeled_count = 0
    updated_count = 0
    
    for idx, row in template_df.iterrows():
        image_path = row['image_path']
        new_disease = row.get('new_disease', '')
        
        # Skip if not labeled
        if not new_disease or new_disease == '':
            continue
            
        labeled_count += 1
        
        # Update in dataset dataframe
        mask = dataset_df['image_path'] == image_path
        if mask.any():
            old_disease = dataset_df.loc[mask, 'disease'].iloc[0]
            dataset_df.loc[mask, 'disease'] = new_disease
            updated_count += 1
            print(f"✅ Updated {image_path}: {old_disease} → {new_disease}")
    
    # Save updated dataset
    dataset_df.to_csv(dataset_path, index=False)
    print(f"\n🎉 Labeling session summary:")
    print(f"  Images reviewed: {len(template_df)}")
    print(f"  Images labeled: {labeled_count}")
    print(f"  Updates applied: {updated_count}")
    print(f"  Dataset saved: {dataset_path}")
    
    # Show new disease distribution
    print(f"\n📊 New disease distribution:")
    disease_counts = dataset_df['disease'].value_counts()
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count}")

def main():
    """
    Main function to update dataset labels
    """
    print("🐔 Update Poultry Disease Labels")
    print("=" * 35)
    
    update_dataset_labels()

if __name__ == "__main__":
    main()