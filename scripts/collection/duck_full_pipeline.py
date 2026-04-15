"""
Duck veri seti uctan uca temizlik pipeline'i.
Goose icin uygulanan ayni konservatif kurallari uygular.

Kullanim:
  python scripts/collection/duck_full_pipeline.py
"""

import sys
import csv
import json
import shutil
import random
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(42)

# ─── Yapilandirma ───
SOURCE_CLASSES = {
    "Bumblefoot": Path("dataset/duck/Bumblefoot"),
    "Fowl_Pox": Path("dataset/duck/Fowl_Pox"),
    "Duck_Plague": Path("dataset/duck/Duck_Plague"),
}

CLEANED_ROOT = Path("cleaned_dataset/duck")
CLEANED_REMOVED = CLEANED_ROOT / "removed"

REVIEW_DIR = Path("review")

SPLIT_OUTPUT = Path("prepared_duck_3class_cleaned")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def compute_hash(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_images(directory):
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def analyze_quality(img_path):
    try:
        file_size_kb = img_path.stat().st_size / 1024
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            arr = np.array(img, dtype=np.float32)
            gray = np.mean(arr, axis=2)
            lap_x = np.diff(gray, axis=1)
            lap_y = np.diff(gray, axis=0)
            blur_score = float(np.var(lap_x) + np.var(lap_y))
            brightness = float(np.mean(gray))

            issues = []
            if w < 50 or h < 50:
                issues.append("cok_kucuk")
            if file_size_kb < 2:
                issues.append("cok_hafif")
            if blur_score < 20:
                issues.append("bulanik")
            if brightness < 15:
                issues.append("cok_karanlik")
            if brightness > 250:
                issues.append("cok_aydinlik")

            return {
                "width": w, "height": h,
                "filesize_kb": round(file_size_kb, 1),
                "blur_score": round(blur_score, 1),
                "brightness": round(brightness, 1),
                "issues": issues,
                "is_low_quality": len(issues) > 0,
                "is_corrupt": False,
            }
    except Exception:
        return {
            "width": 0, "height": 0, "filesize_kb": 0,
            "blur_score": 0, "brightness": 0,
            "issues": ["bozuk"], "is_low_quality": True, "is_corrupt": True,
        }


# ═══════════════════════════════════════════
# ASAMA 1: TEMIZLIK
# ═══════════════════════════════════════════
def stage_1_clean():
    print("=" * 60)
    print("  ASAMA 1: DUCK VERI TEMIZLIGI")
    print("=" * 60)

    # Klasorleri hazirla
    for cn in SOURCE_CLASSES:
        (CLEANED_ROOT / cn).mkdir(parents=True, exist_ok=True)
    CLEANED_REMOVED.mkdir(parents=True, exist_ok=True)

    all_records = []
    global_hashes = {}
    stats = {
        "original": {},
        "cleaned": {},
        "removed": 0,
        "duplicate": 0,
        "low_quality": 0,
    }

    for class_name, source_dir in SOURCE_CLASSES.items():
        cleaned_dir = CLEANED_ROOT / class_name
        if cleaned_dir.exists():
            shutil.rmtree(cleaned_dir)
        cleaned_dir.mkdir(parents=True)

        images = get_images(source_dir)
        stats["original"][class_name] = len(images)
        print(f"\n  {class_name}: {len(images)} gorsel")

        kept = 0
        for img_path in images:
            q = analyze_quality(img_path)
            file_hash = compute_hash(img_path)

            record = {
                "filename": img_path.name,
                "class": class_name,
                "path": str(img_path),
                "width": q["width"],
                "height": q["height"],
                "filesize_kb": q["filesize_kb"],
                "blur_score": q["blur_score"],
                "action": "",
                "reason": "",
            }

            if q["is_corrupt"]:
                record["action"] = "remove"
                record["reason"] = "bozuk_dosya"
                shutil.copy2(img_path, CLEANED_REMOVED / img_path.name)
                stats["removed"] += 1
            elif file_hash in global_hashes:
                record["action"] = "remove"
                record["reason"] = f"duplicate_of_{global_hashes[file_hash]}"
                stats["duplicate"] += 1
            elif "cok_kucuk" in q["issues"] or ("bulanik" in q["issues"] and q["blur_score"] < 10):
                record["action"] = "remove"
                record["reason"] = "dusuk_kalite_ciddi"
                shutil.copy2(img_path, CLEANED_REMOVED / img_path.name)
                stats["removed"] += 1
            else:
                record["action"] = "keep"
                record["reason"] = "gecerli"
                dest = cleaned_dir / img_path.name
                counter = 1
                while dest.exists():
                    dest = cleaned_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                    counter += 1
                shutil.copy2(img_path, dest)
                global_hashes[file_hash] = img_path.name
                kept += 1

            if q["is_low_quality"]:
                stats["low_quality"] += 1

            all_records.append(record)

        stats["cleaned"][class_name] = kept

    # CSV
    csv_path = REVIEW_DIR / "duck_review_manifest.csv"
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "class", "path", "width", "height",
            "filesize_kb", "blur_score", "action", "reason",
        ])
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    # Rapor
    report = {
        "timestamp": datetime.now().isoformat(),
        "original": stats["original"],
        "cleaned": stats["cleaned"],
        "removed": stats["removed"],
        "duplicate": stats["duplicate"],
        "low_quality_flag": stats["low_quality"],
    }
    with open(REVIEW_DIR / "duck_cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    txt_path = REVIEW_DIR / "duck_cleaning_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DUCK VERI TEMIZLIK RAPORU\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"{'':25} {'Orijinal':>10} {'Temiz':>10}\n")
        f.write("-" * 50 + "\n")
        for cn in SOURCE_CLASSES:
            f.write(f"{cn:25} {stats['original'][cn]:>10} {stats['cleaned'][cn]:>10}\n")
        f.write("-" * 50 + "\n")
        orig_total = sum(stats["original"].values())
        clean_total = sum(stats["cleaned"].values())
        f.write(f"{'TOPLAM':25} {orig_total:>10} {clean_total:>10}\n\n")
        f.write(f"Removed: {stats['removed']} | Duplicate: {stats['duplicate']}\n")

        removed = [r for r in all_records if r["action"] == "remove"]
        if removed:
            f.write(f"\nCIKARILAN DOSYALAR ({len(removed)})\n" + "-" * 50 + "\n")
            for r in removed:
                f.write(f"  {r['filename']} | {r['class']} | {r['reason']}\n")

    # Konsol
    print(f"\n  {'='*50}")
    print(f"  TEMIZLIK OZETI")
    print(f"  {'='*50}")
    print(f"  {'':25} {'Orijinal':>10} {'Temiz':>10}")
    print(f"  {'-'*50}")
    for cn in SOURCE_CLASSES:
        print(f"  {cn:25} {stats['original'][cn]:>10} {stats['cleaned'][cn]:>10}")
    print(f"  {'-'*50}")
    print(f"  {'TOPLAM':25} {orig_total:>10} {clean_total:>10}")
    print(f"  Removed: {stats['removed']} | Duplicate: {stats['duplicate']}")

    return stats


# ═══════════════════════════════════════════
# ASAMA 2: SPLIT
# ═══════════════════════════════════════════
def stage_2_split():
    print(f"\n{'='*60}")
    print("  ASAMA 2: DUCK TEMIZ SPLIT")
    print("=" * 60)

    TRAIN_R, VAL_R = 0.70, 0.15

    if SPLIT_OUTPUT.exists():
        shutil.rmtree(SPLIT_OUTPUT)

    classes = {}
    for class_name in SOURCE_CLASSES:
        class_dir = CLEANED_ROOT / class_name
        imgs = get_images(class_dir)
        if imgs:
            # Dedup
            seen = {}
            unique = []
            for img in imgs:
                h = compute_hash(img)
                if h not in seen:
                    seen[h] = True
                    unique.append(img)
            classes[class_name] = unique
            print(f"  {class_name}: {len(unique)} benzersiz gorsel")

    stats = {}
    random.seed(42)

    for class_name, images in classes.items():
        random.shuffle(images)
        n = len(images)
        n_train = max(1, round(n * TRAIN_R))
        n_val = max(1, round(n * VAL_R))
        n_test = max(1, n - n_train - n_val)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():
            dest_dir = SPLIT_OUTPUT / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest_dir / img.name)

        stats[class_name] = {
            "total": n,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

    totals = {"train": 0, "val": 0, "test": 0, "total": 0}
    for s in stats.values():
        for k in totals:
            totals[k] += s[k]

    counts = [s["total"] for s in stats.values()]
    imbalance = round(max(counts) / min(counts), 2) if min(counts) > 0 else float("inf")

    print(f"\n  {'Sinif':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Top':>8}")
    print(f"  {'-'*50}")
    for cn, s in stats.items():
        print(f"  {cn:<20} {s['train']:>8} {s['val']:>8} {s['test']:>8} {s['total']:>8}")
    print(f"  {'-'*50}")
    print(f"  {'TOPLAM':<20} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8} {totals['total']:>8}")
    print(f"  Dengesizlik: {imbalance}x")

    split_report = {
        "dataset_name": "prepared_duck_3class_cleaned",
        "timestamp": datetime.now().isoformat(),
        "seed": 42, "classes": stats, "totals": totals,
        "imbalance_ratio": imbalance,
    }
    with open(SPLIT_OUTPUT / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2, ensure_ascii=False)

    with open(SPLIT_OUTPUT / "split_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"CLEANED DUCK DATASET SPLIT\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dengesizlik: {imbalance}x\n{'='*50}\n\n")
        f.write(f"{'Sinif':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Top':>8}\n")
        f.write(f"{'-'*50}\n")
        for cn, s in stats.items():
            f.write(f"{cn:<20} {s['train']:>8} {s['val']:>8} {s['test']:>8} {s['total']:>8}\n")
        f.write(f"{'-'*50}\n")
        f.write(f"{'TOPLAM':<20} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8} {totals['total']:>8}\n")

    return stats


def main():
    print("*" * 60)
    print("  DUCK UCTAN UCA TEMIZLIK PIPELINE")
    print("*" * 60)

    clean_stats = stage_1_clean()
    split_stats = stage_2_split()

    print(f"\n{'*'*60}")
    print("  DUCK PIPELINE TAMAMLANDI")
    print(f"{'*'*60}")
    print(f"  Temiz dataset: {SPLIT_OUTPUT}/")
    print(f"  git push sonrasi Colab komutu:")
    print(f"    python training/train.py --data-dir {SPLIT_OUTPUT} \\")
    print(f"      --output-dir outputs/duck_cleaned_v2_frozen \\")
    print(f"      --model efficientnet_b0 --freeze-backbone \\")
    print(f"      --epochs 40 --lr 1e-3 --batch-size 8 \\")
    print(f"      --label-smoothing 0.05 --patience 12")


if __name__ == "__main__":
    main()
