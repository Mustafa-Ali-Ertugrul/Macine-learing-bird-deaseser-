
import os
import pandas as pd

# 1. Analyze final_dataset_10_classes directory
data_dir = 'Macine learing (bird deaseser)/final_dataset_10_classes'
print(f"--- Directory Analysis: {data_dir} ---")
if os.path.exists(data_dir):
    total_images = 0
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.casefold().endswith(('.jpg', '.png', '.jpeg'))])
            print(f"{cls}: {count}")
            total_images += count
    print(f"Total images in folders: {total_images}")
else:
    print("Directory not found!")

# 2. Analyze CSVs
csv_files = [
    'Macine learing (bird deaseser)/poultry_labeled.csv',
    'Macine learing (bird deaseser)/poultry_labeled_12k.csv'
]

print("\n--- CSV Analysis ---")
for csv_path in csv_files:
    if os.path.exists(csv_path):
        print(f"\nScanning: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"Total Rows: {len(df)}")
            if 'disease' in df.columns:
                print("Label Distribution:")
                print(df['disease'].value_counts().head(10))
                print(f"Unique Labels: {len(df['disease'].unique())}")
            else:
                print("Column 'disease' not found.")
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"File not found: {csv_path}")
