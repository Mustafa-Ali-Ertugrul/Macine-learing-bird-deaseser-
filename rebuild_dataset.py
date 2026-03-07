#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild Dataset - Data Leakage Free Split

Bu script final_dataset_10_classes/ icerisindeki ORIJINAL goruntulerden
temiz bir train/val/test split olusturur.

Kurallar:
- Sadece orijinal (safe_aug_ ile baslamayan) gorseller kullanilir
- Augmentation sadece train sirasinda on-the-fly yapilacak
- Val ve test setlerine augmented goruntu KONMAZ
- Stratified split: train %70, val %15, test %15

Kullanim:
    python rebuild_dataset.py
"""

import os
import sys
import shutil
import json
import random
from collections import defaultdict

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# === Configuration ===
SOURCE_DIR = 'final_dataset_10_classes'
OUTPUT_DIR = 'clean_dataset_split'
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')


def is_original(filename: str) -> bool:
    """Check if a file is an original (non-augmented) image."""
    return not filename.startswith('safe_aug_')


def collect_originals(source_dir: str) -> dict:
    """
    Collect all original (non-augmented) image paths per class.

    Returns:
        dict: {class_name: [list of file paths]}
    """
    class_images = defaultdict(list)

    classes = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        for filename in sorted(os.listdir(cls_path)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            if is_original(filename):
                class_images[cls].append(os.path.join(cls_path, filename))

    return dict(class_images)


def stratified_split(class_images: dict, train_ratio: float, val_ratio: float,
                     test_ratio: float, seed: int) -> tuple:
    """
    Perform stratified split ensuring each class is proportionally represented.

    For classes with very few images (< 6), special handling ensures
    at least 1 image in train, and distributes remaining to val/test.

    Returns:
        (splits dict, stats dict)
    """
    random.seed(seed)

    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list),
    }

    split_stats = {}

    for cls, paths in sorted(class_images.items()):
        n = len(paths)
        shuffled = paths.copy()
        random.shuffle(shuffled)

        if n == 1:
            splits['train'][cls] = shuffled
            split_stats[cls] = {'total': n, 'train': 1, 'val': 0, 'test': 0}

        elif n == 2:
            splits['train'][cls] = shuffled[:1]
            splits['val'][cls] = shuffled[1:2]
            split_stats[cls] = {'total': n, 'train': 1, 'val': 1, 'test': 0}

        elif n <= 5:
            # Ensure at least 1 in each split where possible
            splits['train'][cls] = shuffled[:max(1, n - 2)]
            splits['val'][cls] = shuffled[max(1, n - 2):max(1, n - 2) + 1]
            splits['test'][cls] = shuffled[max(1, n - 2) + 1:]
            split_stats[cls] = {
                'total': n,
                'train': len(splits['train'][cls]),
                'val': len(splits['val'][cls]),
                'test': len(splits['test'][cls])
            }

        else:
            # Normal split
            n_val = max(1, round(n * val_ratio))
            n_test = max(1, round(n * test_ratio))
            n_train = n - n_val - n_test

            if n_train < 1:
                n_train = 1
                remaining = n - 1
                n_val = remaining // 2
                n_test = remaining - n_val

            splits['train'][cls] = shuffled[:n_train]
            splits['val'][cls] = shuffled[n_train:n_train + n_val]
            splits['test'][cls] = shuffled[n_train + n_val:]

            split_stats[cls] = {
                'total': n,
                'train': n_train,
                'val': n_val,
                'test': len(splits['test'][cls])
            }

    return splits, split_stats


def copy_files(splits: dict, output_dir: str) -> None:
    """Copy files to the output directory structure."""
    if os.path.exists(output_dir):
        print(f"  Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    total_copied = 0

    for split_name in ['train', 'val', 'test']:
        for cls, paths in sorted(splits[split_name].items()):
            dest_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for src_path in paths:
                filename = os.path.basename(src_path)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(src_path, dest_path)
                total_copied += 1

    print(f"  Total files copied: {total_copied}")


def verify_no_leakage(splits: dict) -> bool:
    """Verify there is no data leakage between splits."""
    all_ok = True

    for cls in splits['train'].keys():
        train_files = set(os.path.basename(p) for p in splits['train'].get(cls, []))
        val_files = set(os.path.basename(p) for p in splits['val'].get(cls, []))
        test_files = set(os.path.basename(p) for p in splits['test'].get(cls, []))

        train_val = train_files & val_files
        train_test = train_files & test_files
        val_test = val_files & test_files

        if train_val:
            print(f"  LEAKAGE [{cls}] train-val overlap: {len(train_val)} files")
            all_ok = False
        if train_test:
            print(f"  LEAKAGE [{cls}] train-test overlap: {len(train_test)} files")
            all_ok = False
        if val_test:
            print(f"  LEAKAGE [{cls}] val-test overlap: {len(val_test)} files")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("DATASET REBUILD - DATA LEAKAGE FREE SPLIT")
    print("=" * 60)

    # 1. Check source directory
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Please ensure final_dataset_10_classes/ exists.")
        sys.exit(1)

    # 2. Collect original images
    print(f"\nSource: {SOURCE_DIR}")
    print("Collecting original (non-augmented) images...\n")

    class_images = collect_originals(SOURCE_DIR)

    print(f"  {'Class':<30} {'Original Images':>15}")
    print("  " + "-" * 47)
    total = 0
    for cls, paths in sorted(class_images.items()):
        print(f"  {cls:<30} {len(paths):>13}")
        total += len(paths)
    print("  " + "-" * 47)
    print(f"  {'TOTAL':<30} {total:>13}")

    # 3. Perform stratified split
    print(f"\nSplit ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"Random seed: {RANDOM_SEED}\n")

    splits, stats = stratified_split(
        class_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 4. Print split statistics
    print(f"  {'Class':<30} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("  " + "-" * 56)
    totals = {'total': 0, 'train': 0, 'val': 0, 'test': 0}
    for cls, s in sorted(stats.items()):
        print(f"  {cls:<30} {s['total']:>6} {s['train']:>6} {s['val']:>6} {s['test']:>6}")
        for k in totals:
            totals[k] += s[k]
    print("  " + "-" * 56)
    print(f"  {'TOTAL':<30} {totals['total']:>6} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6}")

    # Warn about small classes
    warnings_found = False
    for cls, s in sorted(stats.items()):
        if s['total'] < 10:
            if not warnings_found:
                print("\nWarnings:")
                warnings_found = True
            print(f"  WARNING: {cls} has only {s['total']} original images!")
            if s['val'] == 0:
                print(f"           -> No validation images for this class.")
            if s['test'] == 0:
                print(f"           -> No test images for this class.")

    # 5. Verify no leakage
    print("\nVerifying no data leakage...")
    if verify_no_leakage(splits):
        print("  PASSED: No data leakage detected!")
    else:
        print("  FAILED: Data leakage found! Aborting.")
        sys.exit(1)

    # 6. Copy files
    print(f"\nCopying files to: {OUTPUT_DIR}/")
    copy_files(splits, OUTPUT_DIR)

    # 7. Save metadata
    metadata = {
        'source_dir': SOURCE_DIR,
        'output_dir': OUTPUT_DIR,
        'random_seed': RANDOM_SEED,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'split_stats': stats,
        'total_original_images': total,
        'classes': sorted(class_images.keys()),
        'num_classes': len(class_images),
        'leakage_check': 'PASSED'
    }

    metadata_path = os.path.join(OUTPUT_DIR, 'split_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Metadata saved to: {metadata_path}")

    # 8. Summary
    print("\n" + "=" * 60)
    print("REBUILD COMPLETE!")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"    train/ - {totals['train']} original images")
    print(f"    val/   - {totals['val']} original images")
    print(f"    test/  - {totals['test']} original images")
    print(f"\n  Augmentation will be applied ON-THE-FLY during training only.")
    print(f"  Val and test sets contain ONLY original images.")
    print("=" * 60)


if __name__ == '__main__':
    main()
