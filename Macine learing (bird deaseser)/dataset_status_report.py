import os
from pathlib import Path
import pandas as pd

# Target configuration
TARGET_DISEASES = [
    'Coccidiosis',
    'Healthy',
    'Salmonella',
    'Newcastle Disease',
    "Marek's Disease",
    'Avian Influenza',
    'Infectious Bronchitis',
    'Infectious Bursal Disease',
    'Histomoniasis',
    'Fowl Pox'
]

TARGET_COUNT = 500

# Directories to scan
SEARCH_DIRS = [
    'final_poultry_dataset_10_classes'
]

# Mapping variations to canonical names
NAME_MAPPING = {
    'coccidiosis': 'Coccidiosis',
    'cocci': 'Coccidiosis',
    'healthy': 'Healthy',
    'salmonella': 'Salmonella',
    'salmo': 'Salmonella',
    'newcastle': 'Newcastle Disease',
    'newcastle disease': 'Newcastle Disease',
    'ncd': 'Newcastle Disease',
    'marek': "Marek's Disease",
    "marek's disease": "Marek's Disease",
    'mareks': "Marek's Disease",
    'avian influenza': 'Avian Influenza',
    'bird flu': 'Avian Influenza',
    'ai': 'Avian Influenza',
    'infectious bronchitis': 'Infectious Bronchitis',
    'ib': 'Infectious Bronchitis',
    'infectious bursal disease': 'Infectious Bursal Disease',
    'ibd': 'Infectious Bursal Disease',
    'gumboro': 'Infectious Bursal Disease',
    'histomoniasis': 'Histomoniasis',
    'blackhead': 'Histomoniasis',
    'fowl pox': 'Fowl Pox',
    'pox': 'Fowl Pox'
}

def scan_directories():
    print("🔍 Scanning directories for poultry disease images...")
    
    found_images = {disease: [] for disease in TARGET_DISEASES}
    
    base_path = Path('.')
    
    for search_dir in SEARCH_DIRS:
        dir_path = base_path / search_dir
        if not dir_path.exists():
            continue
            
        print(f"  Scanning {search_dir}...")
        
        # Walk through directory
        for root, dirs, files in os.walk(dir_path):
            # Check if current folder name matches a disease
            folder_name = Path(root).name.lower()
            
            # Try to match folder name to disease
            matched_disease = None
            
            # Direct match check
            if folder_name in NAME_MAPPING:
                matched_disease = NAME_MAPPING[folder_name]
            else:
                # Partial match check
                for key, val in NAME_MAPPING.items():
                    if key in folder_name:
                        matched_disease = val
                        break
            
            if matched_disease:
                # Collect images in this folder
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        full_path = os.path.join(root, file)
                        found_images[matched_disease].append(full_path)

    return found_images

def generate_report(found_images):
    print("\n" + "="*60)
    print("DATASET STATUS REPORT")
    print("="*60)
    print(f"{'DISEASE':<30} | {'FOUND':<10} | {'TARGET':<10} | {'STATUS':<10}")
    print("-" * 70)
    
    total_found = 0
    needs_download = False
    
    report_data = []
    
    for disease in TARGET_DISEASES:
        count = len(found_images[disease])
        total_found += count
        
        status = "✅ OK" if count >= TARGET_COUNT else f"❌ MISSING {TARGET_COUNT - count}"
        if count < TARGET_COUNT:
            needs_download = True
            
        print(f"{disease:<30} | {count:<10} | {TARGET_COUNT:<10} | {status}")
        
        report_data.append({
            'Disease': disease,
            'Found': count,
            'Target': TARGET_COUNT,
            'Missing': max(0, TARGET_COUNT - count)
        })
        
    print("-" * 70)
    print(f"{'TOTAL':<30} | {total_found:<10} | {TARGET_COUNT * 10:<10}")
    print("="*60)
    
    return report_data

def main():
    found_images = scan_directories()
    report = generate_report(found_images)
    
    # Save detailed report
    df = pd.DataFrame(report)
    df.to_csv('dataset_status.csv', index=False)
    print("\nReport saved to dataset_status.csv")

if __name__ == "__main__":
    main()
