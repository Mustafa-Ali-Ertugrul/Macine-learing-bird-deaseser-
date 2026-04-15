
import pandas as pd
import os

csv_path = 'Macine learing (bird deaseser)/Macine learing (bird deaseser)/poultry_labeled.csv'
if not os.path.exists(csv_path):
    print(f"CSV not found at {csv_path}")
    # Try alternate path
    csv_path = 'Macine learing (bird deaseser)/poultry_labeled.csv'

if os.path.exists(csv_path):
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns)
    if 'disease' in df.columns:
        print("Unique diseases:", df['disease'].unique())
    else:
        print("'disease' column not found")
else:
    print("CSV still not found")
