"""
Dataset Temizleme & Augmentation Pipeline
==========================================
1. Perceptual hashing (pHash) ile intra-split duplicate tespiti ve kaldırma
2. Cross-split leakage tespiti (train<->val<->test) ve kaldırma
3. Sadece train set'e augmentation uygulama
4. Detaylı rapor üretme

Kullanım:
    python dataset_clean_and_augment.py
"""

import os
import sys
import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import imagehash
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# ---------- Windows console encoding fix ----------
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================
SOURCE_DIR = Path(r"final_dataset_split")           # Orijinal veri
TARGET_DIR = Path(r"final_dataset_clean")            # Temizlenmis cikti
REPORT_FILE = Path(r"dataset_cleaning_report.txt")

PHASH_THRESHOLD = 4          # Hamming distance <= bu deger -> near-duplicate
TARGET_PER_CLASS = 1500      # Augmentation sonrasi minimum train goruntu/sinif
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Augmentation pipeline (sadece train icin)
AUGMENT_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
report_lines = []

def log(msg):
    """Print and store for report."""
    print(msg)
    report_lines.append(msg)


def is_image(filepath):
    """Check if file is a supported image."""
    return Path(filepath).suffix.lower() in IMAGE_EXTENSIONS


def compute_phash(img_path, hash_size=8):
    """Compute perceptual hash for an image file."""
    try:
        img = Image.open(img_path).convert('RGB')
        return imagehash.phash(img, hash_size=hash_size)
    except Exception:
        return None


def get_image_files(directory):
    """Get all image file paths in a directory."""
    if not directory.exists():
        return []
    return sorted([
        directory / f for f in os.listdir(directory)
        if is_image(f) and (directory / f).is_file()
    ])


# =============================================================================
# STEP 1: COPY SOURCE TO TARGET
# =============================================================================
def step0_copy_dataset():
    log("\n" + "=" * 70)
    log("ADIM 0: Orijinal veriyi hedef dizine kopyalama")
    log("=" * 70)

    if not SOURCE_DIR.exists():
        log(f"HATA: Kaynak dizin bulunamadi: {SOURCE_DIR}")
        sys.exit(1)

    if TARGET_DIR.exists():
        log(f"  Mevcut hedef dizin siliniyor: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)

    log(f"  Kopyalaniyor: {SOURCE_DIR} -> {TARGET_DIR}")
    shutil.copytree(SOURCE_DIR, TARGET_DIR)
    log("  Kopyalama tamamlandi.")

    # Count images
    for split in ['train', 'val', 'test']:
        split_dir = TARGET_DIR / split
        if not split_dir.exists():
            continue
        classes = sorted([d for d in os.listdir(split_dir) if (split_dir / d).is_dir()])
        total = 0
        for cls in classes:
            count = len(get_image_files(split_dir / cls))
            total += count
        log(f"  {split.upper()}: {total} goruntu, {len(classes)} sinif")


# =============================================================================
# STEP 1: INTRA-SPLIT DUPLICATE DETECTION & REMOVAL
# =============================================================================
def step1_remove_intra_split_duplicates():
    log("\n" + "=" * 70)
    log("ADIM 1: Intra-split duplicate tespiti (pHash)")
    log("=" * 70)

    total_removed = 0
    stats = {}  # split -> class -> removed_count

    for split in ['train', 'val', 'test']:
        split_dir = TARGET_DIR / split
        if not split_dir.exists():
            continue

        log(f"\n  --- {split.upper()} ---")
        stats[split] = {}
        classes = sorted([d for d in os.listdir(split_dir) if (split_dir / d).is_dir()])

        for cls in classes:
            cls_dir = split_dir / cls
            images = get_image_files(cls_dir)

            if len(images) == 0:
                stats[split][cls] = 0
                continue

            # Compute hashes
            hash_to_files = defaultdict(list)
            for img_path in tqdm(images, desc=f"  pHash {split}/{cls}", leave=False):
                h = compute_phash(img_path)
                if h is not None:
                    hash_to_files[h].append(img_path)

            # Find exact hash duplicates first
            removed = 0
            seen_hashes = {}
            files_to_remove = set()

            for h, file_list in hash_to_files.items():
                if len(file_list) > 1:
                    # Keep the first, mark rest for removal
                    for dup_file in file_list[1:]:
                        files_to_remove.add(dup_file)

            # Near-duplicate check (Hamming distance <= threshold)
            all_hashes = list(hash_to_files.keys())
            for i in range(len(all_hashes)):
                for j in range(i + 1, len(all_hashes)):
                    dist = all_hashes[i] - all_hashes[j]
                    if dist <= PHASH_THRESHOLD and dist > 0:
                        # Mark the group with fewer files for removal
                        for f in hash_to_files[all_hashes[j]]:
                            files_to_remove.add(f)

            # Remove duplicates
            for f in files_to_remove:
                if f.exists():
                    f.unlink()
                    removed += 1

            stats[split][cls] = removed
            total_removed += removed

            if removed > 0:
                remaining = len(get_image_files(cls_dir))
                log(f"  {cls}: {removed} duplicate silindi (kalan: {remaining})")

    log(f"\n  TOPLAM intrra-split duplicate silinen: {total_removed}")
    return total_removed


# =============================================================================
# STEP 2: CROSS-SPLIT LEAKAGE DETECTION & REMOVAL
# =============================================================================
def step2_remove_cross_split_leakage():
    log("\n" + "=" * 70)
    log("ADIM 2: Cross-split leakage tespiti (pHash)")
    log("=" * 70)

    total_leaked = 0

    classes = sorted([
        d for d in os.listdir(TARGET_DIR / 'train')
        if (TARGET_DIR / 'train' / d).is_dir()
    ])

    for cls in classes:
        # Compute train hashes
        train_dir = TARGET_DIR / 'train' / cls
        train_images = get_image_files(train_dir)
        train_hashes = {}  # hash -> filepath

        for img_path in tqdm(train_images, desc=f"  Train hash {cls}", leave=False):
            h = compute_phash(img_path)
            if h is not None:
                train_hashes[h] = img_path

        # Check val and test against train
        leaked_in_class = 0
        for split in ['val', 'test']:
            split_cls_dir = TARGET_DIR / split / cls
            if not split_cls_dir.exists():
                continue

            images = get_image_files(split_cls_dir)
            for img_path in tqdm(images, desc=f"  Check {split}/{cls}", leave=False):
                h = compute_phash(img_path)
                if h is None:
                    continue

                # Check exact match
                is_leak = False
                if h in train_hashes:
                    is_leak = True
                else:
                    # Check near-duplicate
                    for train_h in train_hashes:
                        if (h - train_h) <= PHASH_THRESHOLD:
                            is_leak = True
                            break

                if is_leak:
                    img_path.unlink()
                    leaked_in_class += 1

        # Also check val <-> test
        val_dir = TARGET_DIR / 'val' / cls
        test_dir = TARGET_DIR / 'test' / cls
        if val_dir.exists() and test_dir.exists():
            val_images = get_image_files(val_dir)
            val_hashes = {}
            for img_path in val_images:
                h = compute_phash(img_path)
                if h is not None:
                    val_hashes[h] = img_path

            test_images = get_image_files(test_dir)
            for img_path in test_images:
                h = compute_phash(img_path)
                if h is None:
                    continue
                is_leak = False
                if h in val_hashes:
                    is_leak = True
                else:
                    for val_h in val_hashes:
                        if (h - val_h) <= PHASH_THRESHOLD:
                            is_leak = True
                            break
                if is_leak:
                    img_path.unlink()
                    leaked_in_class += 1

        if leaked_in_class > 0:
            log(f"  {cls}: {leaked_in_class} leakage tespit edildi ve silindi")
        total_leaked += leaked_in_class

    log(f"\n  TOPLAM cross-split leakage silinen: {total_leaked}")
    return total_leaked


# =============================================================================
# STEP 3: AUGMENT TRAIN SET ONLY
# =============================================================================
def step3_augment_train_only():
    log("\n" + "=" * 70)
    log("ADIM 3: Sadece TRAIN set augmentation")
    log("=" * 70)
    log(f"  Hedef: Her sinif minimum {TARGET_PER_CLASS} goruntu")

    train_dir = TARGET_DIR / 'train'
    classes = sorted([d for d in os.listdir(train_dir) if (train_dir / d).is_dir()])

    total_augmented = 0

    for cls in classes:
        cls_dir = train_dir / cls
        original_images = get_image_files(cls_dir)
        original_count = len(original_images)

        if original_count == 0:
            log(f"  {cls}: BOS SINIF - atlaniyor")
            continue

        needed = TARGET_PER_CLASS - original_count
        if needed <= 0:
            log(f"  {cls}: Zaten yeterli ({original_count} >= {TARGET_PER_CLASS})")
            continue

        log(f"  {cls}: {original_count} -> {TARGET_PER_CLASS} (+{needed} augmentation)")

        # Augment
        aug_count = 0
        attempts = 0
        max_attempts = needed * 3  # Safety limit

        while aug_count < needed and attempts < max_attempts:
            # Pick a random original image
            src_img_path = random.choice(original_images)
            try:
                img = Image.open(src_img_path).convert('RGB')
                aug_img = AUGMENT_TRANSFORMS(img)

                # Generate unique filename
                aug_name = f"aug_{aug_count:04d}_{src_img_path.name}"
                aug_path = cls_dir / aug_name

                # Avoid collision
                if aug_path.exists():
                    aug_name = f"aug_{aug_count:04d}_{random.randint(1000,9999)}_{src_img_path.name}"
                    aug_path = cls_dir / aug_name

                aug_img.save(aug_path, quality=95)
                aug_count += 1
            except Exception as e:
                pass
            attempts += 1

        total_augmented += aug_count

    log(f"\n  TOPLAM augmented goruntu uretildi: {total_augmented}")
    log(f"  NOT: Val ve Test set'lere augmentation UYGULANMADI.")
    return total_augmented


# =============================================================================
# STEP 4: GENERATE REPORT
# =============================================================================
def step4_generate_report(dup_count, leak_count, aug_count):
    log("\n" + "=" * 70)
    log("ADIM 4: Detayli rapor")
    log("=" * 70)

    # Final counts
    log("\n  --- FINAL DATASET DISTRIBUTION ---")
    grand_total = 0
    for split in ['train', 'val', 'test']:
        split_dir = TARGET_DIR / split
        if not split_dir.exists():
            continue
        classes = sorted([d for d in os.listdir(split_dir) if (split_dir / d).is_dir()])
        log(f"\n  {split.upper()}:")
        split_total = 0
        for cls in classes:
            count = len(get_image_files(split_dir / cls))
            split_total += count
            log(f"    {cls}: {count}")
        log(f"    TOPLAM: {split_total}")
        grand_total += split_total

    log(f"\n  GENEL TOPLAM: {grand_total}")

    # Summary
    log("\n  --- OZET ---")
    log(f"  Intra-split duplicate silinen: {dup_count}")
    log(f"  Cross-split leakage silinen:   {leak_count}")
    log(f"  Augmented goruntu uretilen:    {aug_count}")
    log(f"  Kaynak dizin:  {SOURCE_DIR.resolve()}")
    log(f"  Hedef dizin:   {TARGET_DIR.resolve()}")
    log(f"  Tarih:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Write report to file
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\n  Rapor yazildi: {REPORT_FILE.resolve()}")


# =============================================================================
# STEP 5: FINAL VERIFICATION - No Cross-Split Leakage
# =============================================================================
def step5_verify_no_leakage():
    log("\n" + "=" * 70)
    log("ADIM 5: Final dogrulama - Cross-split leakage kontrolu")
    log("=" * 70)

    classes = sorted([
        d for d in os.listdir(TARGET_DIR / 'train')
        if (TARGET_DIR / 'train' / d).is_dir()
    ])

    total_leaks = 0

    for cls in classes:
        train_dir = TARGET_DIR / 'train' / cls
        # Only hash ORIGINAL images (skip aug_ prefixed)
        train_images = [f for f in get_image_files(train_dir) if not f.name.startswith('aug_')]
        train_hashes = set()

        for img_path in train_images:
            h = compute_phash(img_path)
            if h is not None:
                train_hashes.add(h)

        for split in ['val', 'test']:
            split_dir = TARGET_DIR / split / cls
            if not split_dir.exists():
                continue
            images = get_image_files(split_dir)
            for img_path in images:
                h = compute_phash(img_path)
                if h is None:
                    continue
                for th in train_hashes:
                    if (h - th) <= PHASH_THRESHOLD:
                        total_leaks += 1
                        break

    if total_leaks == 0:
        log("  BASARILI: Cross-split leakage tespit edilmedi!")
    else:
        log(f"  UYARI: {total_leaks} potansiyel leakage hala mevcut!")

    return total_leaks


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 70)
    log("  DATASET TEMIZLEME & AUGMENTATION PIPELINE")
    log(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # Step 0: Copy
    step0_copy_dataset()

    # Step 1: Remove intra-split duplicates
    dup_count = step1_remove_intra_split_duplicates()

    # Step 2: Remove cross-split leakage
    leak_count = step2_remove_cross_split_leakage()

    # Step 3: Augment train only
    aug_count = step3_augment_train_only()

    # Step 4: Report
    step4_generate_report(dup_count, leak_count, aug_count)

    # Step 5: Final verification
    step5_verify_no_leakage()

    # Final report save (with verification results)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    log("\n" + "=" * 70)
    log("  PIPELINE TAMAMLANDI!")
    log("=" * 70)


if __name__ == "__main__":
    main()
