import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
TARGET_COUNT = 500
SOURCE_DIRS = [
    'organized_labeled_dataset',
    'poultry_microscopy',
    'new_diseases_dataset',
    'chicken_disease_dataset',
    'organized_poultry_dataset',
    'poultry_dataset_512x512'
]
OUTPUT_DIR = 'final_dataset_10_classes'

# Disease mapping (same as before)
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

TARGET_DISEASES = sorted(list(set(NAME_MAPPING.values())))

def collect_images():
    print("🔍 Collecting existing images...")
    found_images = {disease: [] for disease in TARGET_DISEASES}
    
    base_path = Path('.')
    
    for search_dir in SOURCE_DIRS:
        dir_path = base_path / search_dir
        if not dir_path.exists():
            continue
            
        for root, dirs, files in os.walk(dir_path):
            folder_name = Path(root).name.lower()
            
            matched_disease = None
            if folder_name in NAME_MAPPING:
                matched_disease = NAME_MAPPING[folder_name]
            else:
                for key, val in NAME_MAPPING.items():
                    if key in folder_name:
                        matched_disease = val
                        break
            
            if matched_disease:
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        found_images[matched_disease].append(os.path.join(root, file))
    
    return found_images

def augment_image(image_path, save_path, prefix):
    """Generate an augmented version of the image"""
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Random augmentation strategy
        strategy = random.choice(['rotate', 'flip', 'color', 'noise', 'zoom'])
        
        if strategy == 'rotate':
            angle = random.choice([90, 180, 270, random.randint(-30, 30)])
            img = img.rotate(angle)
        
        elif strategy == 'flip':
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
        elif strategy == 'color':
            enhancer = random.choice([ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color])
            factor = random.uniform(0.7, 1.3)
            img = enhancer(img).enhance(factor)
            
        elif strategy == 'noise':
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
        elif strategy == 'zoom':
            w, h = img.size
            zoom = random.uniform(0.8, 0.95)
            crop_w, crop_h = int(w * zoom), int(h * zoom)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            img = img.crop((left, top, left + crop_w, top + crop_h))
            img = img.resize((w, h), Image.LANCZOS)
            
        img.save(save_path, quality=95)
        return True
        
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")
        return False

def process_dataset():
    found_images = collect_images()
    
    output_base = Path(OUTPUT_DIR)
    output_base.mkdir(exist_ok=True)
    
    print(f"\n🚀 Starting augmentation to reach {TARGET_COUNT} images per class...")
    
    for disease in TARGET_DISEASES:
        images = found_images[disease]
        current_count = len(images)
        
        print(f"\n📊 {disease}: Found {current_count} images")
        
        disease_dir = output_base / disease
        disease_dir.mkdir(exist_ok=True)
        
        # 1. Copy existing images
        copied_count = 0
        for i, img_path in enumerate(images):
            if i >= TARGET_COUNT: break # Don't copy more than needed if we have excess (optional, but good for balance)
            
            ext = os.path.splitext(img_path)[1]
            new_name = f"{disease.lower().replace(' ', '_')}_{i+1:04d}{ext}"
            shutil.copy2(img_path, disease_dir / new_name)
            copied_count += 1
            
        # 2. Augment if needed
        if copied_count < TARGET_COUNT:
            needed = TARGET_COUNT - copied_count
            print(f"   ⚠️ Missing {needed} images. Augmenting...")
            
            # If we have 0 images, we can't augment
            if copied_count == 0:
                print(f"   ❌ No source images for {disease}! Cannot augment.")
                continue
                
            source_images = list(disease_dir.glob('*'))
            
            pbar = tqdm(total=needed)
            generated = 0
            while generated < needed:
                # Pick a random source image from the ALREADY COPIED ones
                src = random.choice(source_images)
                
                new_name = f"{disease.lower().replace(' ', '_')}_aug_{generated+1:04d}.jpg"
                save_path = disease_dir / new_name
                
                if augment_image(src, save_path, generated):
                    generated += 1
                    pbar.update(1)
            
            pbar.close()
            print(f"   ✅ Generated {generated} augmented images.")
        else:
            print("   ✅ Target count met.")

    print(f"\n✨ Dataset prepared in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    process_dataset()
