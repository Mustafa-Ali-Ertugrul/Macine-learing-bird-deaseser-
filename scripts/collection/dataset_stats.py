"""
Dataset class distribution analiz scripti.
dataset/ altindaki tum species/class klasorlerini tarar ve rapor uretir.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATASET_DIR = "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SKIP_DIRS = {"removed_unrelated", "__pycache__"}
SKIP_FILES = {"process_report.json", "process_report.txt",
              "fowl_pox_integration_report.json", "dataset_stats.json",
              "dataset_stats.txt", "fowl_pox_classify_log.txt"}


def count_images(directory):
    count = 0
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def get_total_size_mb(directory):
    total = 0
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            total += f.stat().st_size
    return round(total / (1024 * 1024), 2)


def main():
    root = Path(DATASET_DIR)
    if not root.exists():
        print(f"[HATA] {root} bulunamadi")
        return

    # Veri toplama
    class_data = []  # [{species, class, count, size_mb}]
    species_totals = defaultdict(int)
    class_totals = defaultdict(int)
    total_images = 0

    # dataset/<species>/<class>/ yapisini tara
    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir():
            continue
        if species_dir.name in SKIP_DIRS:
            continue

        for class_dir in sorted(species_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if class_dir.name in SKIP_DIRS:
                continue

            count = count_images(class_dir)
            size_mb = get_total_size_mb(class_dir)

            class_data.append({
                "species": species_dir.name,
                "class": class_dir.name,
                "image_count": count,
                "size_mb": size_mb,
                "path": str(class_dir),
            })

            species_totals[species_dir.name] += count
            class_totals[class_dir.name] += count
            total_images += count

    # Analiz
    empty_classes = [c for c in class_data if c["image_count"] == 0]
    under_10 = [c for c in class_data if 0 < c["image_count"] < 10]
    under_50 = [c for c in class_data if 0 < c["image_count"] < 50]

    # Dengesizlik analizi (species bazinda)
    imbalance = {}
    for species, total in species_totals.items():
        sp_classes = [c for c in class_data if c["species"] == species and c["image_count"] > 0]
        if len(sp_classes) >= 2:
            counts = [c["image_count"] for c in sp_classes]
            max_c = max(counts)
            min_c = min(counts)
            ratio = round(max_c / min_c, 2) if min_c > 0 else float("inf")
            imbalance[species] = {
                "max_class": max(sp_classes, key=lambda x: x["image_count"])["class"],
                "max_count": max_c,
                "min_class": min(sp_classes, key=lambda x: x["image_count"])["class"],
                "min_count": min_c,
                "ratio": ratio,
            }

    # Konsol ciktisi
    print("=" * 70)
    print("  DATASET CLASS DISTRIBUTION RAPORU")
    print(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for species in sorted(species_totals.keys()):
        sp_classes = sorted(
            [c for c in class_data if c["species"] == species],
            key=lambda x: -x["image_count"]
        )
        sp_total = species_totals[species]

        print(f"\n  {'_'*60}")
        emoji = {"duck": "D", "goose": "G", "chicken": "C", "general": "*"}.get(species, " ")
        print(f"  [{emoji}] {species.upper()} (toplam: {sp_total})")
        print(f"  {'_'*60}")
        print(f"  {'Sinif':<30} {'Gorsel':>8} {'Boyut':>10} {'Oran':>8}")
        print(f"  {'-'*58}")

        for c in sp_classes:
            pct = round(c["image_count"] / sp_total * 100, 1) if sp_total > 0 else 0
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            print(f"  {c['class']:<30} {c['image_count']:>8} {c['size_mb']:>8} MB {pct:>6.1f}%  {bar}")

    # Ozet
    print(f"\n{'='*70}")
    print("  GENEL OZET")
    print(f"{'='*70}")
    print(f"  Toplam gorsel:        {total_images}")
    print(f"  Toplam species:       {len(species_totals)}")
    print(f"  Toplam class kaydi:   {len(class_data)}")
    print()

    print("  Species bazli toplam:")
    for sp, cnt in sorted(species_totals.items(), key=lambda x: -x[1]):
        print(f"    {sp}: {cnt}")
    print()

    print("  Class bazli toplam (tum turler):")
    for cls, cnt in sorted(class_totals.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {cnt}")

    if empty_classes:
        print(f"\n  BOS SINIFLAR ({len(empty_classes)}):")
        for c in empty_classes:
            print(f"    {c['species']}/{c['class']}")

    if under_10:
        print(f"\n  10'DAN AZ GORSEL ({len(under_10)}):")
        for c in sorted(under_10, key=lambda x: x["image_count"]):
            print(f"    {c['species']}/{c['class']}: {c['image_count']}")

    if under_50:
        print(f"\n  50'DEN AZ GORSEL ({len(under_50)}):")
        for c in sorted(under_50, key=lambda x: x["image_count"]):
            print(f"    {c['species']}/{c['class']}: {c['image_count']}")

    if imbalance:
        print(f"\n  DENGESIZLIK ANALIZI:")
        for sp, info in sorted(imbalance.items()):
            print(f"    {sp}: max={info['max_class']}({info['max_count']}) "
                  f"min={info['min_class']}({info['min_count']}) "
                  f"oran={info['ratio']}x")

    # JSON rapor
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": total_images,
            "total_species": len(species_totals),
            "total_class_entries": len(class_data),
            "species_totals": dict(species_totals),
            "class_totals": dict(class_totals),
        },
        "class_distribution": class_data,
        "warnings": {
            "empty_classes": [f"{c['species']}/{c['class']}" for c in empty_classes],
            "under_10_images": [
                {"path": f"{c['species']}/{c['class']}", "count": c["image_count"]}
                for c in sorted(under_10, key=lambda x: x["image_count"])
            ],
            "under_50_images": [
                {"path": f"{c['species']}/{c['class']}", "count": c["image_count"]}
                for c in sorted(under_50, key=lambda x: x["image_count"])
            ],
        },
        "imbalance_analysis": imbalance,
    }

    json_path = Path(DATASET_DIR) / "dataset_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # TXT rapor
    txt_path = Path(DATASET_DIR) / "dataset_stats.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DATASET CLASS DISTRIBUTION RAPORU\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"TOPLAM GORSEL: {total_images}\n\n")

        f.write("SPECIES BAZLI\n")
        f.write(f"{'-'*40}\n")
        for sp, cnt in sorted(species_totals.items(), key=lambda x: -x[1]):
            f.write(f"  {sp}: {cnt}\n")

        f.write(f"\nDETAYLI DAGILIM\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Species':<12} {'Class':<30} {'Count':>8}\n")
        f.write(f"{'-'*60}\n")
        for c in sorted(class_data, key=lambda x: (x["species"], -x["image_count"])):
            f.write(f"{c['species']:<12} {c['class']:<30} {c['image_count']:>8}\n")

        if empty_classes:
            f.write(f"\nBOS SINIFLAR\n")
            for c in empty_classes:
                f.write(f"  {c['species']}/{c['class']}\n")

        if under_10:
            f.write(f"\n10'DAN AZ GORSEL\n")
            for c in sorted(under_10, key=lambda x: x["image_count"]):
                f.write(f"  {c['species']}/{c['class']}: {c['image_count']}\n")

        if under_50:
            f.write(f"\n50'DEN AZ GORSEL\n")
            for c in sorted(under_50, key=lambda x: x["image_count"]):
                f.write(f"  {c['species']}/{c['class']}: {c['image_count']}\n")

        if imbalance:
            f.write(f"\nDENGESIZLIK\n")
            for sp, info in sorted(imbalance.items()):
                f.write(f"  {sp}: max={info['max_class']}({info['max_count']}) "
                        f"min={info['min_class']}({info['min_count']}) "
                        f"oran={info['ratio']}x\n")

    print(f"\n  Raporlar:")
    print(f"    {json_path}")
    print(f"    {txt_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
