import json, os
from pathlib import Path
from collections import defaultdict

ROOT = Path("dataset")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP = {"removed_unrelated", "__pycache__"}

data = []
sp_totals = defaultdict(int)
grand = 0

for sp_dir in sorted(ROOT.iterdir()):
    if not sp_dir.is_dir() or sp_dir.name in SKIP:
        continue
    for cls_dir in sorted(sp_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name in SKIP:
            continue
        cnt = sum(1 for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
        data.append({"species": sp_dir.name, "class": cls_dir.name, "images": cnt})
        sp_totals[sp_dir.name] += cnt
        grand += cnt

# Konsol
print("=" * 55)
print(f"{'Species':<12} {'Class':<28} {'Images':>8}")
print("-" * 55)
for d in sorted(data, key=lambda x: (x["species"], -x["images"])):
    print(f"{d['species']:<12} {d['class']:<28} {d['images']:>8}")
print("-" * 55)
for sp, t in sorted(sp_totals.items(), key=lambda x: -x[1]):
    print(f"{sp:<42} {t:>8}")
print(f"{'TOPLAM':<42} {grand:>8}")

# JSON
report = {"counts": data, "species_totals": dict(sp_totals), "grand_total": grand}
Path("dataset/real_image_counts.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)

# TXT
lines = [f"{'Species':<12} {'Class':<28} {'Images':>8}", "-" * 55]
for d in sorted(data, key=lambda x: (x["species"], -x["images"])):
    lines.append(f"{d['species']:<12} {d['class']:<28} {d['images']:>8}")
lines.append("-" * 55)
for sp, t in sorted(sp_totals.items(), key=lambda x: -x[1]):
    lines.append(f"{sp:<42} {t:>8}")
lines.append(f"{'TOPLAM':<42} {grand:>8}")
Path("dataset/real_image_counts.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nRaporlar: dataset/real_image_counts.json + .txt")
