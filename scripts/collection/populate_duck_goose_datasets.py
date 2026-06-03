#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Populate duck and goose datasets to 10 classes each.
Optimized version:
1. Low web image download limit (3 per class) to avoid DDG rate limiting and hanging.
2. Graceful try/except around DDG search to instantly fall back on chicken dataset copying.
3. Fast execution.
"""

import os
import sys
import shutil
import hashlib
import time
import requests
from pathlib import Path
from PIL import Image

# Import configuration
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DUCK_DISEASE_CLASSES, GOOSE_DISEASE_CLASSES, SPECIES_CONFIG, fix_windows_encoding

# Fix encoding
fix_windows_encoding()

CHICKEN_DIR = Path("final_dataset_10_classes")
DUCK_DIR = Path("duck_dataset_10_classes")
GOOSE_DIR = Path("goose_dataset_10_classes")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TIMEOUT = 5
DELAY = 0.2

# Query templates for DDG Search
QUERY_TEMPLATES = {
    "duck": {
        "Avian_Influenza": ["duck avian influenza disease"],
        "Coccidiosis": ["duck coccidiosis clinical"],
        "Healthy": ["healthy duck photo"],
        "Histomoniasis": ["duck histomoniasis blackhead"],
        "Infectious_Bronchitis": ["duck infectious bronchitis"],
        "Newcastle_Disease": ["duck newcastle disease"],
        "Salmonella": ["duck salmonella clinical"],
    },
    "goose": {
        "Avian_Influenza": ["goose avian influenza disease"],
        "Coccidiosis": ["goose coccidiosis clinical"],
        "Healthy": ["healthy goose photo"],
        "Histomoniasis": ["goose histomoniasis blackhead"],
        "Infectious_Bronchitis": ["goose infectious bronchitis"],
        "Infectious_Bursal_Disease": ["goose infectious bursal disease"],
        "Mareks_Disease": ["goose mareks disease"],
        "Newcastle_Disease": ["goose newcastle disease"],
    }
}

def validate_image(image_path: Path) -> bool:
    """Check if the image is valid and not corrupted."""
    try:
        if image_path.stat().st_size < 1024:
            return False
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
            w, h = img.size
            if w < 32 or h < 32:
                return False
        return True
    except Exception:
        return False

def compute_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_unused_folders(dataset_dir: Path, target_classes: list):
    """Remove directories that are not in the target class list."""
    print(f"\nCleaning up unused folders in {dataset_dir}...")
    if not dataset_dir.exists():
        return
    for item in dataset_dir.iterdir():
        if item.is_dir():
            if item.name not in target_classes and not item.name.startswith("_"):
                print(f"  [-] Removing unused folder: {item.name}")
                shutil.rmtree(item)

def search_and_download(species: str, class_name: str, dest_dir: Path, limit: int = 3):
    """Download images from DuckDuckGo search."""
    try:
        from ddgs import DDGS
    except ImportError:
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    queries = QUERY_TEMPLATES.get(species, {}).get(class_name, [f"{species} {class_name} photo"])
    
    downloaded = 0
    seen_urls = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    }

    print(f"  Searching web for {species}/{class_name}...")
    try:
        with DDGS() as ddgs:
            for query in queries:
                if downloaded >= limit:
                    break
                # Lower max_results to 15 to make it super fast
                results = list(ddgs.images(query, max_results=15, safesearch="moderate", type_image="photo"))
                for res in results:
                    if downloaded >= limit:
                        break
                    url = res.get("image") or res.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)

                    try:
                        r = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
                        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                            h = hashlib.md5(url.encode()).hexdigest()[:8]
                            ext = Path(urlparse(url).path).suffix.lower()
                            if ext not in IMAGE_EXTENSIONS:
                                ext = ".jpg"
                            
                            filename = dest_dir / f"web_{class_name}_{h}{ext}"
                            with open(filename, "wb") as f:
                                for chunk in r.iter_content(8192):
                                    f.write(chunk)
                            
                            if validate_image(filename):
                                downloaded += 1
                                print(f"    [+] Downloaded web image {downloaded}/{limit}: {filename.name}")
                            else:
                                if filename.exists():
                                    filename.unlink()
                        time.sleep(DELAY)
                    except Exception:
                        continue
    except Exception as e:
        print(f"    Search error/rate limit encountered: {e}. Falling back to copy.")

    return downloaded

def supplement_from_chicken(class_name: str, dest_dir: Path, target_total: int = 80):
    """Supplement images from chicken dataset to reach target total."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_images = [f for f in dest_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    existing_count = len(existing_images)
    
    if existing_count >= target_total:
        return 0

    to_copy = target_total - existing_count
    source_dir = CHICKEN_DIR / class_name
    
    if not source_dir.exists():
        print(f"  [WARN] Chicken dataset source dir {source_dir} does not exist.")
        return 0

    chicken_images = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"  Supplementing {to_copy} images from Chicken/{class_name}...")
    
    copied = 0
    # Keep track of hashes in dest_dir to avoid duplicates
    dest_hashes = {compute_hash(f) for f in existing_images}

    for img in chicken_images:
        if copied >= to_copy:
            break
        try:
            h = compute_hash(img)
            if h not in dest_hashes:
                dest_hashes.add(h)
                dest_name = dest_dir / f"chicken_{img.name}"
                shutil.copy2(img, dest_name)
                copied += 1
        except Exception:
            continue
            
    print(f"    [+] Copied {copied} images from chicken dataset.")
    return copied

def populate_species(species: str, target_classes: list):
    print(f"\n{'='*70}")
    print(f"  POPULATING {species.upper()} DATASET TO 10 CLASSES")
    print(f"{'='*70}")
    
    dataset_dir = DUCK_DIR if species == "duck" else GOOSE_DIR
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Clean unused folders
    if target_classes:
        clean_unused_folders(dataset_dir, target_classes)
    
    # 2. Populate each class
    for class_name in target_classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Check current count of valid images
        existing = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        valid_count = sum(1 for f in existing if validate_image(f))
        
        # If folder is empty or has very few images, download from web and supplement
        if valid_count < 10:
            print(f"\nClass: {class_name} (Current valid count: {valid_count})")
            # Try web search first (limit is 3 for fast, safe downloading)
            web_count = search_and_download(species, class_name, class_dir, limit=3)
            # Supplement from chicken
            copied = supplement_from_chicken(class_name, class_dir, target_total=80)
        else:
            print(f"Class: {class_name} already populated with {valid_count} images.")

def print_final_summary():
    print(f"\n{'='*70}")
    print("  FINAL IMAGE COUNT SUMMARY")
    print(f"{'='*70}")
    
    for species, path, classes in [("duck", DUCK_DIR, DUCK_DISEASE_CLASSES), ("goose", GOOSE_DIR, GOOSE_DISEASE_CLASSES)]:
        print(f"\n--- {species.upper()} ---")
        if not path.exists():
            print("Not found")
            continue
        for class_name in classes:
            class_dir = path / class_name
            if class_dir.exists():
                imgs = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
                print(f"  {class_name}: {len(imgs)} images")
            else:
                print(f"  {class_name}: Not created")

if __name__ == "__main__":
    from urllib.parse import urlparse
    populate_species("duck", DUCK_DISEASE_CLASSES)
    populate_species("goose", GOOSE_DISEASE_CLASSES)
    print_final_summary()
