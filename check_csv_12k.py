
import pandas as pd
import os

csv_path = 'Macine learing (bird deaseser)/poultry_labeled_12k.csv'

if os.path.exists(csv_path):
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns)
    if 'disease' in df.columns:
        print("Unique diseases:", df['disease'].unique())
        print("Counts:", df['disease'].value_counts())
    else:
        print("'disease' column not found")
else:
    print(f"CSV not found: {csv_path}")
