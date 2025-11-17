#!/usr/bin/env python3
"""
Resize all dataset images to 512x512 pixels
Maintains aspect ratio with padding
Preserves directory structure
"""

from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import os

# Configuration
CONFIG = {
    'target_size': (512, 512),
    'input_dirs': ['poultry_microscopy', 'organized_poultry_dataset'],
    'output_base': 'poultry_dataset_512x512',
    'padding_color': (0, 0, 0),  # Black padding
    'quality': 95,
    'supported_formats': ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
}


def resize_image_with_padding(image_path: Path, output_path: Path, target_size: tuple):
    """
    Resize image to target size with padding to maintain aspect ratio
    """
    try:
        # Open image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate aspect ratio
            original_width, original_height = img.size
            target_width, target_height = target_size
            
            # Calculate scaling factor (maintain aspect ratio)
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            scale_factor = min(width_ratio, height_ratio)
            
            # Calculate new size
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with padding
            new_img = Image.new('RGB', target_size, CONFIG['padding_color'])
            
            # Calculate position to paste (center the image)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            # Paste resized image onto padded background
            new_img.paste(img_resized, (paste_x, paste_y))
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            new_img.save(output_path, 'JPEG', quality=CONFIG['quality'])
            
            return True
            
    except Exception as e:
        print(f"  ❌ Error processing {image_path.name}: {str(e)[:50]}")
        return False


def collect_images(base_dir: Path):
    """
    Collect all image files from directory and subdirectories
    """
    images = []
    for ext in CONFIG['supported_formats']:
        images.extend(base_dir.rglob(f"*{ext}"))
    return images


def resize_dataset():
    """
    Main function to resize entire dataset
    """
    print("=" * 60)
    print("DATASET IMAGE RESIZER - 512x512")
    print("=" * 60)
    
    output_base = Path(CONFIG['output_base'])
    total_processed = 0
    total_errors = 0
    
    # Process each input directory
    for input_dir_name in CONFIG['input_dirs']:
        input_dir = Path(input_dir_name)
        
        if not input_dir.exists():
            print(f"\n⚠️  Directory not found: {input_dir_name}")
            continue
        
        print(f"\n📁 Processing: {input_dir_name}/")
        
        # Collect all images
        images = collect_images(input_dir)
        print(f"   Found {len(images):,} images")
        
        if len(images) == 0:
            continue
        
        # Process each image
        success_count = 0
        error_count = 0
        
        for img_path in tqdm(images, desc=f"   Resizing {input_dir_name}"):
            # Calculate relative path to preserve structure
            rel_path = img_path.relative_to(input_dir)
            
            # Create output path with .jpg extension
            output_path = output_base / input_dir_name / rel_path.with_suffix('.jpg')
            
            # Skip if already processed
            if output_path.exists():
                continue
            
            # Resize image
            if resize_image_with_padding(img_path, output_path, CONFIG['target_size']):
                success_count += 1
            else:
                error_count += 1
        
        print(f"   ✅ Processed: {success_count:,} images")
        if error_count > 0:
            print(f"   ❌ Errors: {error_count}")
        
        total_processed += success_count
        total_errors += error_count
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Total images processed: {total_processed:,}")
    if total_errors > 0:
        print(f"❌ Total errors: {total_errors}")
    print(f"📁 Output directory: {output_base}/")
    print(f"📏 Image size: {CONFIG['target_size'][0]}x{CONFIG['target_size'][1]} pixels")
    
    # Verify output
    print("\n📊 Output directory structure:")
    if output_base.exists():
        for subdir in sorted(output_base.iterdir()):
            if subdir.is_dir():
                img_count = len(list(subdir.rglob("*.jpg")))
                print(f"   {subdir.name}/: {img_count:,} images")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Check PIL
    try:
        from PIL import Image
    except ImportError:
        print("❌ Pillow not installed. Run: python -m pip install Pillow")
        exit(1)
    
    # Check tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        print("❌ tqdm not installed. Run: python -m pip install tqdm")
        exit(1)
    
    # Run resize
    resize_dataset()
