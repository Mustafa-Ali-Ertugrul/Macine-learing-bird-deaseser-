import os
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
import csv

def find_images(directory):
    """
    Find all image files in directory
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                images.append(os.path.join(root, file))
    
    return images

def organize_poultry_microscopy_images():
    """
    Organize the existing poultry microscopy images into class directories
    """
    print("🐔 Organizing Poultry Microscopy Images")
    print("=" * 40)
    
    # Source directory with existing images
    source_dir = Path("poultry_microscopy/all_images")
    
    # Target directory for organized dataset
    target_dir = Path("organized_poultry_dataset")
    target_dir.mkdir(exist_ok=True)
    
    # Create class directories
    classes = ["healthy", "ib", "ibd", "coccidiosis", "salmonella", "fatty_liver", "histomoniasis", "unclassified"]
    
    for class_name in classes:
        class_dir = target_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    # Find all images
    images = find_images(source_dir)
    print(f"🔍 Found {len(images)} images")
    
    # For now, we'll put all images in the "unclassified" folder
    # Later, you can move them to appropriate class folders or use the labeling tool
    unclassified_dir = target_dir / "unclassified"
    
    for i, image_path in enumerate(tqdm(images, desc="Organizing images")):
        src = Path(image_path)
        dst = unclassified_dir / f"poultry_microscopy_{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    
    # Create CSV file
    csv_path = target_dir / "dataset.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'class', 'filename'])
        
        for class_name in classes:
            class_dir = target_dir / class_name
            class_images = find_images(class_dir)
            
            for image_path in class_images:
                rel_path = Path(image_path).relative_to(target_dir)
                filename = Path(image_path).name
                writer.writerow([str(rel_path).replace('\\', '/'), class_name, filename])
    
    print(f"✅ Organized {len(images)} images to {target_dir}")
    print(f"📄 Dataset CSV: {csv_path}")
    
    return target_dir

def main():
    """
    Main function
    """
    dataset_dir = organize_poultry_microscopy_images()
    
    print(f"\n🎉 Success!")
    print(f"📁 Organized dataset: {dataset_dir}")
    print(f"\nNext steps:")
    print(f"1. Review the images in {dataset_dir}/unclassified")
    print(f"2. Use the HTML labeling tool (poultry_labeling_tool.html) to classify them")
    print(f"3. Move classified images to appropriate class folders")
    print(f"4. Re-run this script to update the dataset.csv file")

if __name__ == "__main__":
    main()