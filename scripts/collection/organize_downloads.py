"""
İndirilen hastalık görsellerini dataset klasörlerine taşır.

Kurallar:
  - Metadata JSON'dan class ve source bilgisi okunur
  - Tür (duck/goose/general) kaynak URL'den tespit edilir
  - Duplicate hash kontrolü yapılır
  - Bozuk görseller loglanır
  - Sonunda detaylı rapor üretilir

Kullanım:
    python scripts/collection/organize_downloads.py
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

# ─────────────────────────────────────────
# Yapılandırma
# ─────────────────────────────────────────
SOURCE_DIR = "downloaded_disease_images"
OUTPUT_DIR = "dataset"

CLASS_MAPPING = {
    "avian_pox": "Fowl_Pox",
    "bumblefoot": "Bumblefoot",
    "goose_parvovirus": "Goose_Parvovirus",
    "duck_plague": "Duck_Plague",
    "duck_viral_hepatitis": "Duck_Viral_Hepatitis",
    "waterfowl_viral_disease": "Waterfowl_Viral_Disease",
}

# URL/kaynak tabanlı tür tespiti
DUCK_KEYWORDS = [
    "duck", "ördek", "duck-plague", "duck-viral", "duck_plague",
    "duck_viral", "duck-foot", "duck-bumble",
]
GOOSE_KEYWORDS = [
    "goose", "kaz", "goose-parvo", "goose_parvo", "derzsy",
    "gosling",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def detect_species(metadata: dict, source_class: str) -> str:
    """Metadata ve class adından tür tespiti yap."""
    # 1. Class adından tespit
    source_lower = source_class.lower()
    if any(kw in source_lower for kw in ["duck_plague", "duck_viral"]):
        return "duck"
    if any(kw in source_lower for kw in ["goose_parvo"]):
        return "goose"

    # 2. Source URL'den tespit
    source_page = (metadata.get("source_page") or "").lower()
    image_url = (metadata.get("image_url") or "").lower()
    notes = (metadata.get("notes") or "").lower()
    combined = f"{source_page} {image_url} {notes}"

    duck_score = sum(1 for kw in DUCK_KEYWORDS if kw in combined)
    goose_score = sum(1 for kw in GOOSE_KEYWORDS if kw in combined)

    if duck_score > goose_score and duck_score > 0:
        return "duck"
    if goose_score > duck_score and goose_score > 0:
        return "goose"

    # 3. Source site'den tespit
    source_site = (metadata.get("source_site") or "").lower()
    if "backyardchickens" in source_site:
        # Bumblefoot thread'leri genelde duck ile ilgili
        if "bumble" in source_lower:
            return "duck"

    return "general"


def compute_file_hash(filepath: Path) -> str:
    """Dosyanın MD5 hash'ini hesapla."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(filepath: Path) -> tuple[bool, str]:
    """Görselin geçerli olup olmadığını kontrol et."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img.load()
            w, h = img.size
            if w < 10 or h < 10:
                return False, f"Çok küçük: {w}x{h}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def find_metadata(image_path: Path) -> dict:
    """Görsel dosyası için eşleşen metadata JSON'ı bul."""
    # Aynı isimle .json uzantılı dosya ara
    stem = image_path.stem
    json_path = image_path.parent / f"{stem}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Klasördeki tüm JSON'ları tara, image_url eşleşmesi ara
    for jf in image_path.parent.glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("class") and image_path.stem.startswith(data["class"]):
                # Hash kısmını karşılaştır
                img_hash_part = image_path.stem.split("_")[-1]
                json_hash_part = jf.stem.split("_")[-1]
                if img_hash_part == json_hash_part:
                    return data
        except Exception:
            continue

    return {}


def main():
    source = Path(SOURCE_DIR)
    output = Path(OUTPUT_DIR)

    if not source.exists():
        print(f"[HATA] Kaynak klasör bulunamadı: {source}")
        sys.exit(1)

    # İstatistikler
    stats = {
        "total_found": 0,
        "total_moved": 0,
        "duplicates_skipped": 0,
        "corrupt_skipped": 0,
        "errors": 0,
    }
    class_counts = defaultdict(int)  # "species/class" -> count
    corrupt_log = []
    duplicate_log = []
    moved_files = []
    seen_hashes = {}  # hash -> destination path

    print("=" * 60)
    print("  GÖRSEL ORGANİZATÖR")
    print("=" * 60)
    print(f"  Kaynak:  {source.absolute()}")
    print(f"  Hedef:   {output.absolute()}")
    print(f"  Sınıflar: {len(CLASS_MAPPING)}")
    print("=" * 60)

    # Tüm görselleri bul
    all_images = []
    for dirpath, _, filenames in os.walk(str(source)):
        for fn in filenames:
            fp = Path(dirpath) / fn
            if fp.suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append(fp)

    stats["total_found"] = len(all_images)
    print(f"\nToplam görsel bulundu: {len(all_images)}")

    # Her görseli işle
    for img_path in sorted(all_images):
        # Kaynak class'ı klasör adından al
        # Yapı: downloaded_disease_images/<class>/<source_site>/<image>
        rel = img_path.relative_to(source)
        parts = rel.parts
        if len(parts) >= 1:
            source_class = parts[0]
        else:
            source_class = "unknown"

        # Metadata oku
        metadata = find_metadata(img_path)
        if not metadata:
            # Metadata yoksa klasör adından oluştur
            metadata = {
                "class": source_class,
                "source_page": "",
                "image_url": "",
                "source_site": parts[1] if len(parts) >= 2 else "",
                "notes": "no metadata found",
            }

        # 1. Bozuk görsel kontrolü
        valid, reason = validate_image(img_path)
        if not valid:
            corrupt_log.append({
                "file": str(img_path),
                "reason": reason,
                "class": source_class,
            })
            stats["corrupt_skipped"] += 1
            print(f"  [BOZUK] {img_path.name}: {reason}")
            continue

        # 2. Duplicate kontrolü
        file_hash = compute_file_hash(img_path)
        if file_hash in seen_hashes:
            duplicate_log.append({
                "file": str(img_path),
                "duplicate_of": str(seen_hashes[file_hash]),
                "hash": file_hash,
            })
            stats["duplicates_skipped"] += 1
            continue

        # 3. Tür tespiti
        species = detect_species(metadata, source_class)

        # 4. Class mapping
        mapped_class = CLASS_MAPPING.get(source_class, source_class)

        # 5. Hedef klasör oluştur
        dest_dir = output / species / mapped_class
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 6. Benzersiz dosya adı oluştur
        ext = img_path.suffix.lower()
        short_hash = file_hash[:10]
        dest_filename = f"{mapped_class}_{species}_{short_hash}{ext}"
        dest_path = dest_dir / dest_filename

        # Çakışma kontrolü
        counter = 1
        while dest_path.exists():
            dest_filename = f"{mapped_class}_{species}_{short_hash}_{counter}{ext}"
            dest_path = dest_dir / dest_filename
            counter += 1

        # 7. Kopyala
        try:
            shutil.copy2(img_path, dest_path)
            seen_hashes[file_hash] = dest_path

            # Metadata JSON'ı da kopyala
            meta_dest = dest_path.with_suffix(".json")
            metadata["organized_to"] = str(dest_path)
            metadata["species_detected"] = species
            metadata["mapped_class"] = mapped_class
            metadata["file_hash"] = file_hash
            metadata["organized_at"] = datetime.now(timezone.utc).isoformat()

            with open(meta_dest, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            key = f"{species}/{mapped_class}"
            class_counts[key] += 1
            stats["total_moved"] += 1

            moved_files.append({
                "source": str(img_path),
                "destination": str(dest_path),
                "species": species,
                "class": mapped_class,
                "hash": file_hash,
            })

            print(f"  [OK] {species}/{mapped_class}/{dest_filename}")

        except Exception as e:
            stats["errors"] += 1
            print(f"  [ERR] {img_path}: {e}")

    # ── Raporlar ──
    print(f"\n{'='*60}")
    print("  ÖZET RAPOR")
    print(f"{'='*60}")

    # Sıralı çıktı
    for category in ["duck", "goose", "general"]:
        items = {k: v for k, v in sorted(class_counts.items()) if k.startswith(f"{category}/")}
        if items:
            print(f"\n  📂 {category.upper()}")
            for key, count in items.items():
                class_name = key.split("/")[1]
                print(f"    {class_name}: {count}")

    print(f"\n{'─'*40}")
    print(f"  Toplam bulunan:     {stats['total_found']}")
    print(f"  Taşınan:            {stats['total_moved']}")
    print(f"  Duplicate atlanan:  {stats['duplicates_skipped']}")
    print(f"  Bozuk atlanan:      {stats['corrupt_skipped']}")
    print(f"  Hata:               {stats['errors']}")

    # process_report.json
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source.absolute()),
        "output_dir": str(output.absolute()),
        "stats": stats,
        "class_counts": dict(class_counts),
        "corrupt_files": corrupt_log,
        "duplicate_files": duplicate_log,
        "moved_files": moved_files,
    }

    report_json_path = output / "process_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 JSON rapor: {report_json_path}")

    # process_report.txt
    report_txt_path = output / "process_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("GÖRSEL ORGANİZASYON RAPORU\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")

        f.write("SINIF DAĞILIMI\n")
        f.write(f"{'-'*50}\n")
        for category in ["duck", "goose", "general"]:
            items = {k: v for k, v in sorted(class_counts.items()) if k.startswith(f"{category}/")}
            if items:
                f.write(f"\n[{category.upper()}]\n")
                for key, count in items.items():
                    class_name = key.split("/")[1]
                    f.write(f"  {class_name}: {count}\n")

        f.write(f"\n{'='*50}\n")
        f.write(f"Toplam bulunan:     {stats['total_found']}\n")
        f.write(f"Taşınan:            {stats['total_moved']}\n")
        f.write(f"Duplicate atlanan:  {stats['duplicates_skipped']}\n")
        f.write(f"Bozuk atlanan:      {stats['corrupt_skipped']}\n")
        f.write(f"Hata:               {stats['errors']}\n")

        if corrupt_log:
            f.write(f"\n{'='*50}\n")
            f.write("BOZUK DOSYALAR\n")
            for item in corrupt_log:
                f.write(f"  {item['file']}: {item['reason']}\n")

        if duplicate_log:
            f.write(f"\n{'='*50}\n")
            f.write("DUPLICATE DOSYALAR\n")
            for item in duplicate_log:
                f.write(f"  {item['file']}\n")
                f.write(f"    -> duplicate of: {item['duplicate_of']}\n")

    print(f"  📄 TXT rapor: {report_txt_path}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
