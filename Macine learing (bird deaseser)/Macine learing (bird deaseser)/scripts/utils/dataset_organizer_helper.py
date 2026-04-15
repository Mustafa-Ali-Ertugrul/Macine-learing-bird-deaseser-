import os
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
import csv

def update_dataset_csv(dataset_dir):
    """
    Update the dataset CSV file with current folder structure
    """
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "dataset.csv"
    
    # Find all class directories
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name != 'unclassified']
    
    # Add unclassified as a class
    class_dirs.append(dataset_dir / 'unclassified')
    
    # Create/update CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'class', 'filename'])
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            # Find all images in this class directory
            for image_path in class_dir.glob('*'):
                if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    rel_path = image_path.relative_to(dataset_dir)
                    writer.writerow([str(rel_path).replace('\\', '/'), class_name, image_path.name])
    
    print(f"[Success] Updated dataset CSV: {csv_path}")

def move_images_by_class(source_dir, target_dir, class_name):
    """
    Move images from source directory to target class directory
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir) / class_name
    target_dir.mkdir(exist_ok=True)
    
    # Move all images from source to target
    moved_count = 0
    for image_path in source_dir.glob('*'):
        if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            dst = target_dir / image_path.name
            shutil.move(str(image_path), str(dst))
            moved_count += 1
    
    print(f"[Success] Moved {moved_count} images to {target_dir}")

def main():
    """
    Main function to help with organizing the dataset
    """
    print("[Chicken] Poultry Disease Dataset Organizer Helper")
    print("=" * 45)
    
    dataset_dir = Path("organized_poultry_dataset")
    unclassified_dir = dataset_dir / "unclassified"
    
    if not unclassified_dir.exists():
        print("[Error] Unclassified directory not found!")
        return
    
    # Count images in unclassified
    unclassified_images = list(unclassified_dir.glob('*'))
    print(f"[Stats] Images in unclassified: {len(unclassified_images)}")
    
    # Show class options
    classes = ["healthy", "ib", "ibd", "coccidiosis", "salmonella", "fatty_liver", "histomoniasis"]
    print(f"\n[Folder] Available classes:")
    for i, class_name in enumerate(classes, 1):
        class_dir = dataset_dir / class_name
        class_images = list(class_dir.glob('*'))
        print(f"  {i}. {class_name} ({len(class_images)} images)")
    
    print(f"  8. Update dataset CSV only")
    print(f"  0. Exit")
    
    while True:
        try:
            choice = input(f"\nSelect an option (0-8): ").strip()
            
            if choice == "0":
                print("[Exit] Exiting...")
                break
            elif choice == "8":
                update_dataset_csv(dataset_dir)
                break
            elif choice in ["1", "2", "3", "4", "5", "6", "7"]:
                class_index = int(choice) - 1
                class_name = classes[class_index]
                
                print(f"\n[Move] Moving images to '{class_name}' class")
                print("How many images to move?")
                print("1. All images")
                print("2. Specific number")
                
                move_choice = input("Select option (1-2): ").strip()
                
                if move_choice == "1":
                    move_images_by_class(unclassified_dir, dataset_dir, class_name)
                    update_dataset_csv(dataset_dir)
                elif move_choice == "2":
                    try:
                        num_images = int(input("Enter number of images to move: "))
                        # Move first N images
                        target_dir = dataset_dir / class_name
                        target_dir.mkdir(exist_ok=True)
                        
                        moved_count = 0
                        for image_path in unclassified_dir.glob('*'):
                            if moved_count >= num_images:
                                break
                            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                                dst = target_dir / image_path.name
                                shutil.move(str(image_path), str(dst))
                                moved_count += 1
                        
                        print(f"[Success] Moved {moved_count} images to {target_dir}")
                        update_dataset_csv(dataset_dir)
                    except ValueError:
                        print("[Error] Invalid number")
                else:
                    print("[Error] Invalid option")
            else:
                print("[Error] Invalid option")
                
        except KeyboardInterrupt:
            print("\n[Exit] Exiting...")
            break
        except Exception as e:
            print(f"[Error] Error: {e}")

if __name__ == "__main__":
    main()