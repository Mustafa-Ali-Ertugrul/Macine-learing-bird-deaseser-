"""
Yerel Fowl_Pox koleksiyonlarini ana dataset'e entegre eder.

Kaynak:
  - duck_dataset_10_classes/Fowl_Pox/     -> dataset/duck/Fowl_Pox/
  - goose_dataset_10_classes/Fowl_Pox/    -> dataset/goose/Fowl_Pox/
  - final_dataset_10_classes/Fowl_Pox/    -> dataset/chicken/Fowl_Pox/

Kurallar:
  - Hash bazli duplicate kontrolu
  - Bozuk gorsel tespiti
  - Her gorsel icin metadata JSON
"""

import json
import hashlib
import shutil
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from PIL import Image

SOURCES = [
    {
        "name": "duck",
        "source_dir": "duck_dataset_10_classes/Fowl_Pox",
        "dest_dir": "dataset/duck/Fowl_Pox",
        "species": "duck",
    },
    {
        "name": "goose",
        "source_dir": "goose_dataset_10_classes/Fowl_Pox",
        "dest_dir": "dataset/goose/Fowl_Pox",
        "species": "goose",
    },
    {
        "name": "chicken",
        "source_dir": "final_dataset_10_classes/Fowl_Pox",
        "dest_dir": "dataset/chicken/Fowl_Pox",
        "species": "chicken",
    },
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def compute_hash(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img.load()
            w, h = img.size
            if w < 10 or h < 10:
                return False, f"too small {w}x{h}", 0, 0
            return True, "OK", w, h
    except Exception as e:
        return False, str(e), 0, 0


def main():
    global_hashes = {}
    stats = {
        "total_found": 0,
        "total_copied": 0,
        "duplicates_skipped": 0,
        "corrupt_removed": 0,
    }
    species_counts = {}
    corrupt_log = []
    duplicate_log = []

    print("=" * 60)
    print("  FOWL_POX YEREL KOLEKSIYON ENTEGRASYONU")
    print("=" * 60)

    # Oncelikle hedef klasorlerdeki mevcut gorsellerin hash'lerini topla
    for src_cfg in SOURCES:
        dest = Path(src_cfg["dest_dir"])
        if dest.exists():
            for f in dest.iterdir():
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    h = compute_hash(f)
                    global_hashes[h] = str(f)

    print(f"  Mevcut dataset'teki gorsel sayisi: {len(global_hashes)}")
    print()

    for src_cfg in SOURCES:
        species = src_cfg["species"]
        source = Path(src_cfg["source_dir"])
        dest = Path(src_cfg["dest_dir"])

        print(f"{'_'*60}")
        print(f"  {species.upper()}: {source} -> {dest}")
        print(f"{'_'*60}")

        if not source.exists():
            print(f"  [UYARI] Kaynak klasor bulunamadi: {source}")
            species_counts[species] = 0
            continue

        dest.mkdir(parents=True, exist_ok=True)

        images = [f for f in sorted(source.iterdir())
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        stats["total_found"] += len(images)
        print(f"  Kaynak gorsel sayisi: {len(images)}")

        copied = 0
        dupes = 0
        corrupt = 0

        for idx, img_path in enumerate(images, start=1):
            # 1. Bozuk kontrol
            valid, reason, w, h = validate_image(img_path)
            if not valid:
                corrupt += 1
                stats["corrupt_removed"] += 1
                corrupt_log.append({
                    "file": str(img_path),
                    "species": species,
                    "reason": reason,
                })
                print(f"  [BOZUK] {img_path.name}: {reason}")
                continue

            # 2. Duplicate kontrol
            file_hash = compute_hash(img_path)
            if file_hash in global_hashes:
                dupes += 1
                stats["duplicates_skipped"] += 1
                duplicate_log.append({
                    "file": str(img_path),
                    "duplicate_of": global_hashes[file_hash],
                    "species": species,
                })
                continue

            # 3. Benzersiz dosya adi
            ext = img_path.suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"
            short_hash = file_hash[:10]
            dest_filename = f"Fowl_Pox_{species}_{idx:04d}_{short_hash}{ext}"
            dest_path = dest / dest_filename

            counter = 1
            while dest_path.exists():
                dest_filename = f"Fowl_Pox_{species}_{idx:04d}_{short_hash}_{counter}{ext}"
                dest_path = dest / dest_filename
                counter += 1

            # 4. Kopyala
            shutil.copy2(img_path, dest_path)
            global_hashes[file_hash] = str(dest_path)

            # 5. Metadata JSON
            metadata = {
                "class": "Fowl_Pox",
                "species": species,
                "source": "local_collection",
                "source_file": str(img_path),
                "file_hash": file_hash,
                "dimensions": f"{w}x{h}",
                "file_size_kb": round(img_path.stat().st_size / 1024, 1),
                "needs_manual_review": False,
                "integrated_at": datetime.now(timezone.utc).isoformat(),
            }

            meta_path = dest_path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            copied += 1
            stats["total_copied"] += 1

        species_counts[species] = copied
        print(f"  Kopyalanan: {copied}")
        print(f"  Duplicate atlanan: {dupes}")
        print(f"  Bozuk: {corrupt}")

    # Final sayim - hedef klasorlerdeki toplam
    final_counts = {}
    for src_cfg in SOURCES:
        species = src_cfg["species"]
        dest = Path(src_cfg["dest_dir"])
        if dest.exists():
            count = len([f for f in dest.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
            final_counts[species] = count
        else:
            final_counts[species] = 0

    # Rapor
    print(f"\n{'='*60}")
    print("  SONUC RAPORU")
    print(f"{'='*60}")
    print(f"\n  Yeni eklenen gorseller:")
    for species, count in species_counts.items():
        print(f"    {species}/Fowl_Pox: +{count}")

    print(f"\n  Toplam gorsel (hedef klasorlerde):")
    for species, count in final_counts.items():
        print(f"    {species}/Fowl_Pox: {count}")

    print(f"\n  Istatistikler:")
    print(f"    Toplam bulunan:     {stats['total_found']}")
    print(f"    Kopyalanan:         {stats['total_copied']}")
    print(f"    Duplicate atlanan:  {stats['duplicates_skipped']}")
    print(f"    Bozuk:              {stats['corrupt_removed']}")

    # JSON rapor
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "species_new_counts": species_counts,
        "species_final_counts": final_counts,
        "stats": stats,
        "corrupt_files": corrupt_log,
        "duplicate_files": duplicate_log,
    }

    report_path = Path("dataset/fowl_pox_integration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Rapor: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
