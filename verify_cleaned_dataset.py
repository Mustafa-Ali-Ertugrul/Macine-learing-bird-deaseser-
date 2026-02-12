"""
Temizlenmis Dataset Dogrulama Script'i
=======================================
final_dataset_clean/ uzerinde:
1. Her split/sinif goruntu sayisi kontrolu
2. Cross-split leakage kontrolu (pHash)
3. Augmented goruntuler sadece train'de mi kontrolu

Kullanim:
    python verify_cleaned_dataset.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image
from tqdm import tqdm

# Windows console encoding fix
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATASET_DIR = Path(r"final_dataset_clean")
PHASH_THRESHOLD = 4
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def get_image_files(directory):
    if not directory.exists():
        return []
    return sorted([
        directory / f for f in os.listdir(directory)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS and (directory / f).is_file()
    ])


def main():
    log("=" * 70)
    log("  DATASET DOGRULAMA RAPORU")
    log("=" * 70)

    if not DATASET_DIR.exists():
        log(f"HATA: Dizin bulunamadi: {DATASET_DIR}")
        return

    # =====================================================
    # CHECK 1: Image counts per class per split
    # =====================================================
    log("\n--- CHECK 1: Goruntu Sayilari ---")
    all_pass = True

    for split in ['train', 'val', 'test']:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            log(f"  UYARI: {split}/ dizini yok!")
            all_pass = False
            continue

        classes = sorted([d for d in os.listdir(split_dir) if (split_dir / d).is_dir()])
        log(f"\n  {split.upper()} ({len(classes)} sinif):")
        total = 0
        for cls in classes:
            count = len(get_image_files(split_dir / cls))
            total += count
            marker = "  " if count > 0 else "!!"
            log(f"    {marker} {cls}: {count}")
            if count == 0:
                all_pass = False
        log(f"    TOPLAM: {total}")

    log(f"\n  Sonuc: {'BASARILI' if all_pass else 'SORUNLAR VAR'}")

    # =====================================================
    # CHECK 2: Augmented images only in train
    # =====================================================
    log("\n--- CHECK 2: Augmented Goruntular Sadece Train'de ---")
    aug_in_wrong_split = 0

    for split in ['val', 'test']:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        classes = sorted([d for d in os.listdir(split_dir) if (split_dir / d).is_dir()])
        for cls in classes:
            files = get_image_files(split_dir / cls)
            for f in files:
                if f.name.startswith('aug_') or f.name.startswith('safe_aug_'):
                    aug_in_wrong_split += 1

    if aug_in_wrong_split == 0:
        log("  BASARILI: Val/Test setlerinde augmented goruntu yok.")
    else:
        log(f"  HATA: Val/Test setlerinde {aug_in_wrong_split} augmented goruntu bulundu!")

    # =====================================================
    # CHECK 3: Cross-split leakage (pHash)
    # =====================================================
    log("\n--- CHECK 3: Cross-Split Leakage Kontrolu (pHash) ---")

    classes = sorted([
        d for d in os.listdir(DATASET_DIR / 'train')
        if (DATASET_DIR / 'train' / d).is_dir()
    ])

    total_leaks = 0

    for cls in classes:
        # Hash ORIGINAL train images only (skip aug_ prefix)
        train_dir = DATASET_DIR / 'train' / cls
        train_images = [f for f in get_image_files(train_dir) if not f.name.startswith('aug_')]
        train_hashes = set()

        for img_path in tqdm(train_images, desc=f"  Train hash {cls}", leave=False):
            try:
                img = Image.open(img_path).convert('RGB')
                h = imagehash.phash(img)
                train_hashes.add(h)
            except Exception:
                continue

        # Check val and test
        class_leaks = 0
        for split in ['val', 'test']:
            split_dir = DATASET_DIR / split / cls
            if not split_dir.exists():
                continue
            images = get_image_files(split_dir)
            for img_path in tqdm(images, desc=f"  Check {split}/{cls}", leave=False):
                try:
                    img = Image.open(img_path).convert('RGB')
                    h = imagehash.phash(img)
                    for th in train_hashes:
                        if (h - th) <= PHASH_THRESHOLD:
                            class_leaks += 1
                            break
                except Exception:
                    continue

        if class_leaks > 0:
            log(f"  UYARI: {cls} -> {class_leaks} potansiyel leakage!")
        total_leaks += class_leaks

    if total_leaks == 0:
        log("  BASARILI: Cross-split leakage tespit edilmedi!")
    else:
        log(f"  UYARI: Toplam {total_leaks} potansiyel leakage!")

    # =====================================================
    # SUMMARY
    # =====================================================
    log("\n" + "=" * 70)
    log("  GENEL SONUC")
    log("=" * 70)

    checks_passed = all_pass and (aug_in_wrong_split == 0) and (total_leaks == 0)
    if checks_passed:
        log("  TUM KONTROLLER BASARILI!")
    else:
        log("  BAZI KONTROLLER BASARISIZ - Yukaridaki detaylari inceleyin.")

    # Save report
    report_file = Path("dataset_verification_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\n  Rapor yazildi: {report_file.resolve()}")


if __name__ == "__main__":
    main()
