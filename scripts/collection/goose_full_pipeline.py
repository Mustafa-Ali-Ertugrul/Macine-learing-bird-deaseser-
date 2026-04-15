"""
Goose veri seti uctan uca temizlik + yeniden hazirlama pipeline'i.

Kullanim:
  python scripts/collection/goose_full_pipeline.py

Asamalar:
  1 - Veri temizligi (dusuk kalite/bozuk cikar, sinif koruma)
  2 - Temizlenmis dataset split
  3 - Egitim komutlari uret (Colab icin)
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
SOURCE_FOWL = Path("dataset/goose/Fowl_Pox")
SOURCE_PARVO = Path("dataset/goose/Goose_Parvovirus")

CLEANED_ROOT = Path("cleaned_dataset/goose")
CLEANED_FOWL = CLEANED_ROOT / "Fowl_Pox"
CLEANED_PARVO = CLEANED_ROOT / "Goose_Parvovirus"
CLEANED_UNCERTAIN = CLEANED_ROOT / "uncertain"
CLEANED_REMOVED = CLEANED_ROOT / "removed"

REVIEW_DIR = Path("review")
SUSPECT_DIR = REVIEW_DIR / "goose_suspect"
LOW_QUALITY_DIR = REVIEW_DIR / "goose_low_quality"

SPLIT_OUTPUT = Path("prepared_goose_2class_cleaned")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Jenerik isim kaliplari
GENERIC_PATTERNS = [
    r"^images?\s*(\(\d+\))?\.jpe?g$",
    r"^indir\s*(\(\d+\))?\.jpe?g$",
    r"^download\s*(\(\d+\))?\.jpe?g$",
    r"^img_?\d*\.jpe?g$",
    r"^photo\s*(\(\d+\))?\.jpe?g$",
    r"^\d+\.jpe?g$",
]


def is_generic_name(filename):
    for pattern in GENERIC_PATTERNS:
        if re.match(pattern, filename.lower(), re.IGNORECASE):
            return True
    return False


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
    except Exception as e:
        return {
            "width": 0, "height": 0, "filesize_kb": 0,
            "blur_score": 0, "brightness": 0,
            "issues": ["bozuk"], "is_low_quality": True, "is_corrupt": True,
        }


# ═══════════════════════════════════════════════════
# ASAMA 1: VERI TEMIZLIGI
# ═══════════════════════════════════════════════════
def stage_1_clean():
    print("=" * 60)
    print("  ASAMA 1: VERI TEMIZLIGI")
    print("=" * 60)

    # Klasor hazirla
    for d in [CLEANED_FOWL, CLEANED_PARVO, CLEANED_UNCERTAIN, CLEANED_REMOVED,
              SUSPECT_DIR, LOW_QUALITY_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    all_records = []
    hashes_seen = {}
    stats = {
        "original": {"Fowl_Pox": 0, "Goose_Parvovirus": 0},
        "cleaned": {"Fowl_Pox": 0, "Goose_Parvovirus": 0},
        "removed": 0, "uncertain": 0, "duplicate": 0,
        "suspicious_name": 0, "low_quality": 0,
    }

    for class_name, source_dir, cleaned_dir in [
        ("Fowl_Pox", SOURCE_FOWL, CLEANED_FOWL),
        ("Goose_Parvovirus", SOURCE_PARVO, CLEANED_PARVO),
    ]:
        images = get_images(source_dir)
        stats["original"][class_name] = len(images)
        print(f"\n  {class_name}: {len(images)} gorsel")

        for img_path in images:
            q = analyze_quality(img_path)
            is_suspect = is_generic_name(img_path.name)
            file_hash = compute_hash(img_path)

            record = {
                "filename": img_path.name,
                "class": class_name,
                "path": str(img_path),
                "width": q["width"],
                "height": q["height"],
                "filesize_kb": q["filesize_kb"],
                "blur_score": q["blur_score"],
                "brightness": q["brightness"],
                "suspicious_name": is_suspect,
                "is_low_quality": q["is_low_quality"],
                "quality_issues": ", ".join(q["issues"]),
                "action": "",
                "reason": "",
            }

            if is_suspect:
                stats["suspicious_name"] += 1
            if q["is_low_quality"]:
                stats["low_quality"] += 1

            # ─── Karar mantigi ───
            # KONSERVATIF: sadece bariz sorunlulari cikar

            # 1. Bozuk dosya -> remove
            if q["is_corrupt"]:
                record["action"] = "remove"
                record["reason"] = "bozuk_dosya"
                shutil.copy2(img_path, CLEANED_REMOVED / img_path.name)
                stats["removed"] += 1

            # 2. Duplicate -> skip
            elif file_hash in hashes_seen:
                record["action"] = "remove"
                record["reason"] = f"duplicate_of_{hashes_seen[file_hash]}"
                stats["duplicate"] += 1

            # 3. Cok kucuk veya bulanik -> remove
            elif "cok_kucuk" in q["issues"] or ("bulanik" in q["issues"] and q["blur_score"] < 10):
                record["action"] = "remove"
                record["reason"] = "dusuk_kalite_ciddi"
                shutil.copy2(img_path, CLEANED_REMOVED / img_path.name)
                shutil.copy2(img_path, LOW_QUALITY_DIR / img_path.name)
                stats["removed"] += 1

            # 4. Gecerli gorsel -> sinifinda tut
            else:
                record["action"] = "keep"
                record["reason"] = "gecerli"
                dest = cleaned_dir / img_path.name
                counter = 1
                while dest.exists():
                    dest = cleaned_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                    counter += 1
                shutil.copy2(img_path, dest)
                hashes_seen[file_hash] = img_path.name
                stats["cleaned"][class_name] += 1

                # Supheli isim olsa bile kopyala (sadece NOT olarak isaretle)
                if is_suspect:
                    shutil.copy2(img_path, SUSPECT_DIR / img_path.name)

            all_records.append(record)

    # CSV manifest
    csv_path = REVIEW_DIR / "goose_review_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "class", "path", "width", "height", "filesize_kb",
            "blur_score", "brightness", "suspicious_name", "is_low_quality",
            "quality_issues", "action", "reason",
        ])
        writer.writeheader()
        for r in sorted(all_records, key=lambda x: (x["action"] != "remove", x["class"])):
            writer.writerow(r)

    # Raporlar
    report = {
        "timestamp": datetime.now().isoformat(),
        "original": stats["original"],
        "cleaned": stats["cleaned"],
        "removed": stats["removed"],
        "uncertain": stats["uncertain"],
        "duplicate": stats["duplicate"],
        "suspicious_name_count": stats["suspicious_name"],
        "low_quality_count": stats["low_quality"],
        "records": all_records,
    }

    json_path = REVIEW_DIR / "goose_cleaning_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    txt_path = REVIEW_DIR / "goose_cleaning_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("GOOSE VERI TEMIZLIK RAPORU\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"{'':30} {'Orijinal':>10} {'Temiz':>10}\n")
        f.write("-" * 55 + "\n")
        for cn in ["Fowl_Pox", "Goose_Parvovirus"]:
            f.write(f"{cn:30} {stats['original'][cn]:>10} {stats['cleaned'][cn]:>10}\n")
        f.write("-" * 55 + "\n")
        orig_total = sum(stats["original"].values())
        clean_total = sum(stats["cleaned"].values())
        f.write(f"{'TOPLAM':30} {orig_total:>10} {clean_total:>10}\n\n")

        f.write(f"Removed:          {stats['removed']}\n")
        f.write(f"Uncertain:        {stats['uncertain']}\n")
        f.write(f"Duplicate:        {stats['duplicate']}\n")
        f.write(f"Suspicious names: {stats['suspicious_name']} (egitimde tutuldu)\n")
        f.write(f"Low quality flag: {stats['low_quality']}\n\n")

        removed = [r for r in all_records if r["action"] == "remove"]
        if removed:
            f.write("CIKARILAN DOSYALAR\n")
            f.write("-" * 55 + "\n")
            for r in removed:
                f.write(f"  {r['filename']} | {r['class']} | {r['reason']}\n")

    # Konsol ozet
    print(f"\n  {'='*50}")
    print(f"  TEMIZLIK OZETI")
    print(f"  {'='*50}")
    print(f"  {'':25} {'Orijinal':>10} {'Temiz':>10}")
    print(f"  {'-'*50}")
    for cn in ["Fowl_Pox", "Goose_Parvovirus"]:
        print(f"  {cn:25} {stats['original'][cn]:>10} {stats['cleaned'][cn]:>10}")
    print(f"  {'-'*50}")
    print(f"  {'TOPLAM':25} {orig_total:>10} {clean_total:>10}")
    print(f"  Removed: {stats['removed']} | Duplicate: {stats['duplicate']}")
    print(f"  Supheli isimli (tutuldu): {stats['suspicious_name']}")

    return stats


# ═══════════════════════════════════════════════════
# ASAMA 2: TEMIZLENMIS DATASET SPLIT
# ═══════════════════════════════════════════════════
def stage_2_split():
    print(f"\n{'='*60}")
    print("  ASAMA 2: TEMIZLENMIS DATASET SPLIT")
    print("=" * 60)

    TRAIN_R, VAL_R, TEST_R = 0.70, 0.15, 0.15

    if SPLIT_OUTPUT.exists():
        shutil.rmtree(SPLIT_OUTPUT)

    classes = {}
    for class_name, class_dir in [("Fowl_Pox", CLEANED_FOWL),
                                   ("Goose_Parvovirus", CLEANED_PARVO)]:
        imgs = get_images(class_dir)
        if not imgs:
            print(f"  [UYARI] {class_dir} bos!")
            continue

        # Dedup
        seen = {}
        unique = []
        for img in imgs:
            h = compute_hash(img)
            if h not in seen:
                seen[h] = img
                unique.append(img)
        classes[class_name] = unique
        print(f"  {class_name}: {len(unique)} benzersiz gorsel")

    if not classes:
        print("  [HATA] Temiz dataset bos!")
        return None

    stats = {}
    random.seed(42)

    for class_name, images in classes.items():
        random.shuffle(images)
        n = len(images)
        n_train = max(1, round(n * TRAIN_R))
        n_val = max(1, round(n * VAL_R))
        n_test = max(1, n - n_train - n_val)
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test

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

    # Rapor
    totals = {"train": 0, "val": 0, "test": 0, "total": 0}
    for s in stats.values():
        for k in totals:
            totals[k] += s[k]

    # Imbalance
    counts = [s["total"] for s in stats.values()]
    imbalance = round(max(counts) / min(counts), 2) if min(counts) > 0 else float("inf")

    print(f"\n  {'Sinif':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Top':>8}")
    print(f"  {'-'*56}")
    for cn, s in stats.items():
        print(f"  {cn:<25} {s['train']:>8} {s['val']:>8} {s['test']:>8} {s['total']:>8}")
    print(f"  {'-'*56}")
    print(f"  {'TOPLAM':<25} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8} {totals['total']:>8}")
    print(f"  Dengesizlik: {imbalance}x")

    # Eski ile karsilastir
    old_path = Path("prepared_goose_2class/split_stats.json")
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        print(f"\n  Eski: {old['totals']['total']} | Yeni: {totals['total']} | Fark: {totals['total']-old['totals']['total']:+d}")

    # Kaydet
    split_report = {
        "dataset_name": "prepared_goose_2class_cleaned",
        "timestamp": datetime.now().isoformat(),
        "seed": 42,
        "split_ratios": {"train": TRAIN_R, "val": VAL_R, "test": TEST_R},
        "classes": stats,
        "totals": totals,
        "imbalance_ratio": imbalance,
    }
    with open(SPLIT_OUTPUT / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2, ensure_ascii=False)

    txt_path = SPLIT_OUTPUT / "split_stats.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("CLEANED GOOSE DATASET SPLIT\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dengesizlik: {imbalance}x\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"{'Sinif':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Top':>8}\n")
        f.write("-" * 56 + "\n")
        for cn, s in stats.items():
            f.write(f"{cn:<25} {s['train']:>8} {s['val']:>8} {s['test']:>8} {s['total']:>8}\n")
        f.write("-" * 56 + "\n")
        f.write(f"{'TOPLAM':<25} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8} {totals['total']:>8}\n")

    return stats


# ═══════════════════════════════════════════════════
# ASAMA 3: EGITIM KOMUTLARI
# ═══════════════════════════════════════════════════
def stage_3_training_commands():
    print(f"\n{'='*60}")
    print("  ASAMA 3: COLAB EGITIM KOMUTLARI")
    print("=" * 60)

    commands = """
# ════════════════════════════════════════════
# GOOSE CLEANED - COLAB EGITIM KOMUTLARI
# ════════════════════════════════════════════

# 1. Repo guncelle
!cd Macine-learing-bird-deaseser- && git pull

# ────────────────────────────
# DENEY 1: Frozen baseline
# ────────────────────────────
!cd Macine-learing-bird-deaseser- && python training/train.py \\
    --data-dir prepared_goose_2class_cleaned \\
    --output-dir outputs/goose_cleaned_v2_frozen \\
    --model efficientnet_b0 \\
    --freeze-backbone \\
    --epochs 40 --lr 1e-3 --batch-size 8 \\
    --label-smoothing 0.05 --patience 12

# ────────────────────────────
# DENEY 2: Partial unfreeze
# ────────────────────────────
!cd Macine-learing-bird-deaseser- && python training/train.py \\
    --data-dir prepared_goose_2class_cleaned \\
    --output-dir outputs/goose_cleaned_v3_unfreeze \\
    --model efficientnet_b0 \\
    --freeze-backbone --unfreeze-last-n 2 \\
    --epochs 30 --lr 1e-4 --batch-size 8 \\
    --label-smoothing 0.05 --patience 10 \\
    --optimizer adamw --weight-decay 1e-4

# ────────────────────────────
# DENEY 3: ResNet18
# ────────────────────────────
!cd Macine-learing-bird-deaseser- && python training/train.py \\
    --data-dir prepared_goose_2class_cleaned \\
    --output-dir outputs/goose_cleaned_resnet18 \\
    --model resnet18 \\
    --freeze-backbone --unfreeze-last-n 1 \\
    --epochs 30 --lr 5e-4 --batch-size 8 \\
    --patience 10

# ────────────────────────────
# SONUC KARSILASTIRMA
# ────────────────────────────
import json, os, csv

experiments = [
    "goose_v2_frozen", "goose_v3_unfreeze", "goose_resnet18",
    "goose_cleaned_v2_frozen", "goose_cleaned_v3_unfreeze", "goose_cleaned_resnet18",
]

results = []
print(f"{'Deney':<35} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
print("-" * 70)

for exp in experiments:
    path = f"Macine-learing-bird-deaseser-/outputs/{exp}/metrics.json"
    dtype = "cleaned" if "cleaned" in exp else "original"
    if os.path.exists(path):
        m = json.load(open(path))
        print(f"{exp:<35} {m['accuracy']:>7.2f}% {m['macro_f1']:>7.2f}% "
              f"{m['macro_precision']:>7.2f}% {m['macro_recall']:>7.2f}%")
        results.append({"experiment": exp, "dataset": dtype, **m})
    else:
        print(f"{exp:<35} {'--':>8} {'--':>8}")

# En iyi model
if results:
    best = max(results, key=lambda x: (x["macro_f1"], x["accuracy"]))
    print(f"\\nEN IYI MODEL: {best['experiment']}")
    print(f"  Macro F1: {best['macro_f1']}% | Accuracy: {best['accuracy']}%")

    # Karsilastirma kaydet
    with open("Macine-learing-bird-deaseser-/outputs/goose_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # En iyi modeli kopyala
    best_src = f"Macine-learing-bird-deaseser-/outputs/{best['experiment']}"
    best_dst = "Macine-learing-bird-deaseser-/outputs/goose_best_final"
    os.makedirs(best_dst, exist_ok=True)
    for fn in ["best_model.pth", "final_summary.json", "confusion_matrix.png",
               "classification_report.txt", "metrics.json"]:
        src = f"{best_src}/{fn}"
        if os.path.exists(src):
            import shutil; shutil.copy2(src, f"{best_dst}/{fn}")
    print(f"  En iyi model kopyalandi: {best_dst}/")
"""

    # Colab komutlarini dosyaya kaydet
    colab_path = Path("training/goose_cleaned_colab_commands.py")
    with open(colab_path, "w", encoding="utf-8") as f:
        f.write(commands)

    print(f"  Colab komutlari kaydedildi: {colab_path}")
    print(f"\n  Bu komutlari Google Colab'a kopyala ve calistir.")
    print(f"  Oncelikle 'git push' yapmayi unutma!")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    print("*" * 60)
    print("  GOOSE UCTAN UCA TEMIZLIK + EGITIM PIPELINE")
    print("*" * 60)

    # Asama 1
    clean_stats = stage_1_clean()

    # Asama 2
    split_stats = stage_2_split()

    # Asama 3
    stage_3_training_commands()

    # Final ozet
    print(f"\n{'*'*60}")
    print("  PIPELINE OZETI")
    print(f"{'*'*60}")
    print(f"\n  Asama 1 - Temizlik:")
    print(f"    Orijinal: {sum(clean_stats['original'].values())} gorsel")
    print(f"    Temiz:    {sum(clean_stats['cleaned'].values())} gorsel")
    print(f"    Removed:  {clean_stats['removed']}")
    print(f"  Asama 2 - Split: {SPLIT_OUTPUT}/")
    print(f"  Asama 3 - Colab komutlari hazir")
    print(f"\n  SONRAKI ADIMLAR:")
    print(f"  1. git add + commit + push")
    print(f"  2. Colab'da egitimi baslat")
    print(f"  3. Sonuclari karsilastir")
    print(f"{'*'*60}")


if __name__ == "__main__":
    main()
