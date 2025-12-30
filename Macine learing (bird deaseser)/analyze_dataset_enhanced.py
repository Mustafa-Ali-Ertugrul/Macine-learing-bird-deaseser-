import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

def analyze_dataset(dataset_dir, output_file="dataset_analysis.csv"):
    """
    Comprehensive dataset analysis
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        print(f"❌ Directory not found: {dataset_dir}")
        return None
    
    results = []
    
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\nAnalyzing: {class_name}")
        
        image_files = list(class_dir.glob("*.*"))
        
        for img_path in tqdm(image_files, desc=f"  Processing"):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                    format = img.format
                    
                    results.append({
                        'image_path': str(img_path),
                        'class': class_name,
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height if height > 0 else 0,
                        'mode': mode,
                        'format': format,
                        'file_size_mb': img_path.stat().st_size / (1024 * 1024)
                    })
            except Exception as e:
                print(f"  ❌ Error reading {img_path.name}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'class': class_name,
                    'width': 0,
                    'height': 0,
                    'aspect_ratio': 0,
                    'mode': 'error',
                    'format': 'error',
                    'file_size_mb': 0
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Analysis saved to {output_file}")
    print_summary(df)
    
    return df

def print_summary(df):
    """Print dataset summary statistics"""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal images: {len(df)}")
    print(f"Valid images: {len(df[df['width'] > 0])}")
    print(f"Corrupted images: {len(df[df['width'] == 0])}")
    
    print("\n📊 Class Distribution:")
    class_counts = df['class'].value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(df) * 100) if len(df) > 0 else 0
        print(f"  {class_name:25} : {count:5} ({percentage:5.1f}%)")
    
    print("\n📐 Image Size Statistics:")
    valid_df = df[df['width'] > 0]
    print(f"  Width  - Min: {valid_df['width'].min()}, Max: {valid_df['width'].max()}, Avg: {valid_df['width'].mean():.0f}")
    print(f"  Height - Min: {valid_df['height'].min()}, Max: {valid_df['height'].max()}, Avg: {valid_df['height'].mean():.0f}")
    
    print("\n🎨 Image Format Distribution:")
    print(df['format'].value_counts())
    
    print("\n💾 File Size Statistics (MB):")
    print(f"  Min: {valid_df['file_size_mb'].min():.2f}, Max: {valid_df['file_size_mb'].max():.2f}, Avg: {valid_df['file_size_mb'].mean():.2f}")

def main():
    print("🔍 Dataset Analyzer")
    print("=" * 60)
    
    dataset_dir = "final_dataset_10_classes"
    output_file = "dataset_analysis.csv"
    
    df = analyze_dataset(dataset_dir, output_file)
    
    if df is not None:
        print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
