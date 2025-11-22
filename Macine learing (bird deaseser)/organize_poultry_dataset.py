import os
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

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

def organize_images_by_class(source_dir, target_dir):
    """
    Organize images by class (disease type) into subdirectories
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for class directories in source
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    total_images = 0
    
    if class_dirs:
        print("📁 Found class directories. Organizing by class...")
        # Organize by existing class structure
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_target = target_dir / class_name
            class_target.mkdir(exist_ok=True)
            
            images = find_images(class_dir)
            total_images += len(images)
            
            for i, image_path in enumerate(tqdm(images, desc=f"Organizing {class_name}")):
                src = Path(image_path)
                dst = class_target / f"{class_name}_{i:04d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
    else:
        print("📁 No class directories found. Organizing all images into a single folder...")
        # All images in one directory - put them in a single folder
        images = find_images(source_dir)
        total_images = len(images)
        
        # Create a single directory for all images
        all_target = target_dir / "all_images"
        all_target.mkdir(exist_ok=True)
        
        for i, image_path in enumerate(tqdm(images, desc="Organizing images")):
            src = Path(image_path)
            dst = all_target / f"poultry_{i:04d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
    
    print(f"✅ Organized {total_images} images to {target_dir}")
    return total_images

def create_dataset_csv(dataset_dir, csv_path):
    """
    Create a CSV file with image paths and classes
    """
    import csv
    
    dataset_dir = Path(dataset_dir)
    csv_path = Path(csv_path)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'class', 'filename'])
        
        # Look for class directories
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        if class_dirs:
            # Multi-class dataset
            for class_dir in class_dirs:
                class_name = class_dir.name
                images = find_images(class_dir)
                
                for image_path in images:
                    rel_path = Path(image_path).relative_to(dataset_dir)
                    filename = Path(image_path).name
                    writer.writerow([str(rel_path).replace('\\', '/'), class_name, filename])
        else:
            # Single directory dataset
            images = find_images(dataset_dir)
            for image_path in images:
                rel_path = Path(image_path).relative_to(dataset_dir)
                filename = Path(image_path).name
                writer.writerow([str(rel_path).replace('\\', '/'), 'unknown', filename])
    
    print(f"✅ Created dataset CSV: {csv_path}")

def main():
    """
    Main function to organize poultry disease datasets
    """
    print("🐔 Poultry Disease Dataset Organizer")
    print("=" * 35)
    
    # Check if user has placed a dataset in raw_dataset folder
    raw_dataset_dir = Path("raw_dataset")
    
    if not raw_dataset_dir.exists():
        print("❌ 'raw_dataset' folder not found!")
        print("📁 Please create a folder named 'raw_dataset' and place your downloaded poultry disease dataset in it.")
        return
    
    if not any(raw_dataset_dir.iterdir()):
        print("❌ 'raw_dataset' folder is empty!")
        print("📁 Please place your downloaded poultry disease dataset in the 'raw_dataset' folder.")
        return
    
    print("🔍 Found raw dataset folder. Organizing images...")
    
    # Organize images
    organized_dir = Path("organized_poultry_dataset")
    total_images = organize_images_by_class(raw_dataset_dir, organized_dir)
    
    # Create CSV file
    csv_path = organized_dir / "dataset.csv"
    create_dataset_csv(organized_dir, csv_path)
    
    print(f"\n🎉 Success!")
    print(f"📁 Organized dataset: {organized_dir}")
    print(f"📊 Total images: {total_images}")
    print(f"📄 Dataset CSV: {csv_path}")
    print(f"\nNext steps:")
    print(f"1. Review the organized dataset in {organized_dir}")
    print(f"2. Use the HTML labeling tool to annotate any unclassified images")
    print(f"3. Train your poultry disease classification model")

if __name__ == "__main__":
    main()