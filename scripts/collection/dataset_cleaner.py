"""
Dataset temizlik araçları:
  1. Bozuk (corrupt) görselleri tespit et ve sil
  2. Duplicate görselleri perceptual hash ile bul ve sil
  3. Çok küçük görselleri ele

Kullanım:
    python scripts/collection/dataset_cleaner.py --dir duck_dataset_10_classes
    python scripts/collection/dataset_cleaner.py --dir duck_dataset_10_classes --delete
    python scripts/collection/dataset_cleaner.py --dir duck_dataset_10_classes --delete --min-size 50

Kurulum:
    pip install Pillow imagehash
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from collections import defaultdict

from PIL import Image

# imagehash opsiyonel — yoksa MD5 hash kullanılır
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("[UYARI] imagehash bulunamadı, MD5 hash kullanılacak.")
    print("        Daha iyi duplicate tespiti için: pip install imagehash")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


def find_images(root_dir: str) -> list[Path]:
    """Tüm görsel dosyalarını bul."""
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(dirpath) / fn)
    return sorted(images)


def check_corrupt(image_path: Path) -> bool:
    """Görselin bozuk olup olmadığını kontrol et."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # verify() sonrası tekrar aç (verify dosyayı kapatır)
        with Image.open(image_path) as img:
            img.load()
        return False  # bozuk değil
    except Exception:
        return True  # bozuk


def check_too_small(image_path: Path, min_size: int = 50) -> bool:
    """Görselin çok küçük olup olmadığını kontrol et."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            return w < min_size or h < min_size
    except Exception:
        return False


def compute_hash(image_path: Path) -> str:
    """Görselin hash'ini hesapla (perceptual veya MD5)."""
    if HAS_IMAGEHASH:
        try:
            with Image.open(image_path) as img:
                return str(imagehash.phash(img))
        except Exception:
            pass

    # Fallback: MD5
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(images: list[Path]) -> list[tuple[Path, Path]]:
    """Duplicate görselleri bul."""
    hash_map: dict[str, Path] = {}
    duplicates = []

    for img_path in images:
        h = compute_hash(img_path)
        if h in hash_map:
            duplicates.append((hash_map[h], img_path))
        else:
            hash_map[h] = img_path

    return duplicates


def main():
    parser = argparse.ArgumentParser(description="Dataset temizlik aracı")
    parser.add_argument("--dir", required=True, help="Taranacak dataset klasörü")
    parser.add_argument("--delete", action="store_true", help="Bozuk/duplicate dosyaları sil")
    parser.add_argument("--min-size", type=int, default=50, help="Minimum görsel boyutu (px)")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"[HATA] Klasör bulunamadı: {root}")
        sys.exit(1)

    print(f"Taranan klasör: {root}")
    images = find_images(str(root))
    print(f"Toplam görsel: {len(images)}\n")

    # ── 1. Bozuk görseller ──
    print("=" * 50)
    print("1. BOZUK GÖRSEL KONTROLÜ")
    print("=" * 50)
    corrupt_files = []
    for img in images:
        if check_corrupt(img):
            corrupt_files.append(img)
            print(f"  [BOZUK] {img}")

    if not corrupt_files:
        print("  Bozuk görsel bulunamadı ✅")
    else:
        print(f"\n  Toplam bozuk: {len(corrupt_files)}")
        if args.delete:
            for f in corrupt_files:
                f.unlink()
                print(f"  [SİLİNDİ] {f}")

    # ── 2. Çok küçük görseller ──
    print(f"\n{'='*50}")
    print(f"2. ÇOK KÜÇÜK GÖRSEL KONTROLÜ (< {args.min_size}px)")
    print("=" * 50)
    small_files = []
    for img in images:
        if img in corrupt_files:
            continue
        if check_too_small(img, args.min_size):
            small_files.append(img)
            try:
                with Image.open(img) as im:
                    w, h = im.size
                print(f"  [KÜÇÜK] {img} ({w}x{h})")
            except Exception:
                print(f"  [KÜÇÜK] {img}")

    if not small_files:
        print(f"  {args.min_size}px altında görsel yok ✅")
    else:
        print(f"\n  Toplam küçük: {len(small_files)}")
        if args.delete:
            for f in small_files:
                f.unlink()
                print(f"  [SİLİNDİ] {f}")

    # ── 3. Duplicate görseller ──
    print(f"\n{'='*50}")
    print("3. DUPLICATE GÖRSEL KONTROLÜ")
    print("=" * 50)

    remaining = [img for img in images if img not in corrupt_files and img not in small_files]
    duplicates = find_duplicates(remaining)

    if not duplicates:
        print("  Duplicate bulunamadı ✅")
    else:
        print(f"  {len(duplicates)} duplicate çift bulundu:")
        for orig, dup in duplicates:
            print(f"  [DUP] {dup}")
            print(f"        orijinal: {orig}")

        if args.delete:
            for _, dup in duplicates:
                dup.unlink()
                print(f"  [SİLİNDİ] {dup}")

    # ── Özet ──
    print(f"\n{'='*50}")
    print("ÖZET")
    print("=" * 50)

    # Sınıf bazlı sayılar
    class_counts: dict[str, int] = defaultdict(int)
    for img in remaining:
        if img not in [d[1] for d in duplicates]:
            class_name = img.parent.name
            class_counts[class_name] += 1

    print(f"  Toplam görsel:    {len(images)}")
    print(f"  Bozuk:            {len(corrupt_files)}")
    print(f"  Çok küçük:        {len(small_files)}")
    print(f"  Duplicate:        {len(duplicates)}")
    print(f"  Temiz kalan:      {len(images) - len(corrupt_files) - len(small_files) - len(duplicates)}")

    if class_counts:
        print(f"\n  Sınıf bazlı dağılım:")
        for cls, count in sorted(class_counts.items()):
            print(f"    {cls}: {count}")


if __name__ == "__main__":
    main()
