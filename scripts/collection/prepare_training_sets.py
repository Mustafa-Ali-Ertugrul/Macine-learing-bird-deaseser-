"""
Egitim veri setleri hazirlayici.
Stratified train/val/test split uygular, duplicate kontrolu yapar.

Cikti:
  prepared_duck_3class/  (Bumblefoot, Fowl_Pox, Duck_Plague)
  prepared_goose_2class/ (Fowl_Pox, Goose_Parvovirus)
"""

import json
import hashlib
import shutil
import random
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DATASETS = [
    {
        "name": "prepared_duck_3class",
        "classes": {
            "Bumblefoot": "dataset/duck/Bumblefoot",
            "Fowl_Pox": "dataset/duck/Fowl_Pox",
            "Duck_Plague": "dataset/duck/Duck_Plague",
        },
    },
    {
        "name": "prepared_goose_2class",
        "classes": {
            "Fowl_Pox": "dataset/goose/Fowl_Pox",
            "Goose_Parvovirus": "dataset/goose/Goose_Parvovirus",
        },
    },
]


def compute_hash(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_images(directory):
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def deduplicate(images):
    seen = {}
    unique = []
    dupes = []
    for img in images:
        h = compute_hash(img)
        if h in seen:
            dupes.append((img, seen[h]))
        else:
            seen[h] = img
            unique.append(img)
    return unique, dupes


def stratified_split(images, train_r, val_r, test_r):
    random.shuffle(images)
    n = len(images)
    n_train = max(1, round(n * train_r))
    n_val = max(1, round(n * val_r))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    return train, val, test


def prepare_dataset(config):
    name = config["name"]
    output_root = Path(name)

    print(f"\n{'='*60}")
    print(f"  DATASET: {name}")
    print(f"{'='*60}")

    # Temiz baslangic
    if output_root.exists():
        shutil.rmtree(output_root)

    stats = {
        "dataset_name": name,
        "seed": SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "classes": {},
        "totals": {"train": 0, "val": 0, "test": 0, "total": 0},
        "duplicates_removed": 0,
    }

    all_dupes = []
    split_totals = {"train": 0, "val": 0, "test": 0}

    random.seed(SEED)

    for class_name, source_dir in config["classes"].items():
        print(f"\n  Sinif: {class_name}")
        print(f"  Kaynak: {source_dir}")

        # 1. Gorselleri topla
        images = get_images(source_dir)
        print(f"  Toplam gorsel: {len(images)}")

        # 2. Duplicate tespiti
        unique_images, dupes = deduplicate(images)
        if dupes:
            print(f"  Duplicate cikarilan: {len(dupes)}")
            all_dupes.extend(dupes)
            stats["duplicates_removed"] += len(dupes)
        print(f"  Benzersiz gorsel: {len(unique_images)}")

        # 3. Stratified split
        train, val, test = stratified_split(unique_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

        print(f"  Split: train={len(train)} val={len(val)} test={len(test)}")

        # 4. Kopyala
        for split_name, split_images in [("train", train), ("val", val), ("test", test)]:
            dest_dir = output_root / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                dest = dest_dir / img.name
                # Isim cakismasi kontrolu
                counter = 1
                while dest.exists():
                    stem = img.stem
                    dest = dest_dir / f"{stem}_{counter}{img.suffix}"
                    counter += 1
                shutil.copy2(img, dest)

            split_totals[split_name] += len(split_images)

        # Sinif istatistikleri
        stats["classes"][class_name] = {
            "source": source_dir,
            "total_source": len(images),
            "duplicates": len(dupes),
            "unique": len(unique_images),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    stats["totals"] = {
        "train": split_totals["train"],
        "val": split_totals["val"],
        "test": split_totals["test"],
        "total": sum(split_totals.values()),
    }

    # Dengesizlik orani
    class_counts = [s["unique"] for s in stats["classes"].values()]
    if class_counts and min(class_counts) > 0:
        stats["imbalance_ratio"] = round(max(class_counts) / min(class_counts), 2)
    else:
        stats["imbalance_ratio"] = float("inf")

    stats["timestamp"] = datetime.now().isoformat()

    # JSON rapor
    json_path = output_root / "split_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # TXT rapor
    txt_path = output_root / "split_stats.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"DATASET: {name}\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Split: train={TRAIN_RATIO} val={VAL_RATIO} test={TEST_RATIO}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"{'Sinif':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Toplam':>8}\n")
        f.write(f"{'-'*60}\n")
        for cls, s in stats["classes"].items():
            total = s["train"] + s["val"] + s["test"]
            f.write(f"{cls:<25} {s['train']:>8} {s['val']:>8} {s['test']:>8} {total:>8}\n")
        f.write(f"{'-'*60}\n")
        t = stats["totals"]
        f.write(f"{'TOPLAM':<25} {t['train']:>8} {t['val']:>8} {t['test']:>8} {t['total']:>8}\n")
        f.write(f"\nDengesizlik orani: {stats['imbalance_ratio']}x\n")
        f.write(f"Cikarilan duplicate: {stats['duplicates_removed']}\n")

    # Konsol ozet
    print(f"\n  {'─'*50}")
    print(f"  {'Sinif':<25} {'Train':>6} {'Val':>6} {'Test':>6} {'Top':>6}")
    print(f"  {'─'*50}")
    for cls, s in stats["classes"].items():
        total = s["train"] + s["val"] + s["test"]
        print(f"  {cls:<25} {s['train']:>6} {s['val']:>6} {s['test']:>6} {total:>6}")
    print(f"  {'─'*50}")
    t = stats["totals"]
    print(f"  {'TOPLAM':<25} {t['train']:>6} {t['val']:>6} {t['test']:>6} {t['total']:>6}")
    print(f"  Dengesizlik: {stats['imbalance_ratio']}x")
    print(f"  Raporlar: {json_path}, {txt_path}")

    return stats


def main():
    print("=" * 60)
    print("  EGITIM VERI SETI HAZIRLAYICI")
    print("=" * 60)

    all_stats = []
    for ds_config in DATASETS:
        s = prepare_dataset(ds_config)
        all_stats.append(s)

    print(f"\n{'='*60}")
    print("  GENEL OZET")
    print(f"{'='*60}")
    for s in all_stats:
        t = s["totals"]
        print(f"\n  {s['dataset_name']}:")
        print(f"    Train: {t['train']} | Val: {t['val']} | Test: {t['test']} | Toplam: {t['total']}")
        print(f"    Siniflar: {len(s['classes'])} | Dengesizlik: {s['imbalance_ratio']}x")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
