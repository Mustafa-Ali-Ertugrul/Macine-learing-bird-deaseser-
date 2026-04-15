#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yetersiz görüntüsü olan sınıflar için veri çoğaltma.

Kullanım:
    python augment_dataset.py --species goose --min-per-class 50
    python augment_dataset.py --species duck --min-per-class 100 --target-class Healthy
    python augment_dataset.py --species goose --min-per-class 50 --dry-run
"""

import os
import sys
import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    SPECIES_CONFIG, SUPPORTED_SPECIES, DISEASE_CLASSES,
    check_dataset_exists,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augment_image(image: Image.Image, aug_index: int) -> Image.Image:
    """Tek bir görüntüye çeşitli augmentation uygula."""
    augmentations = [
        # 0: Yatay çevirme
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        # 1: Döndürme (rastgele açı)
        lambda img: img.rotate(random.uniform(-30, 30), fillcolor=(0, 0, 0)),
        # 2: Parlaklık değiştirme
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)),
        # 3: Kontrast değiştirme
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)),
        # 4: Renk doygunluğu
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
        # 5: Hafif bulanıklık
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))),
        # 6: Keskinleştirme
        lambda img: img.filter(ImageFilter.SHARPEN),
        # 7: Yatay çevirme + parlaklık
        lambda img: ImageEnhance.Brightness(
            img.transpose(Image.FLIP_LEFT_RIGHT)
        ).enhance(random.uniform(0.8, 1.2)),
        # 8: Döndürme + kontrast
        lambda img: ImageEnhance.Contrast(
            img.rotate(random.uniform(-20, 20), fillcolor=(0, 0, 0))
        ).enhance(random.uniform(0.8, 1.2)),
        # 9: Kırpma + yeniden boyutlandırma
        lambda img: _random_crop_resize(img),
    ]

    aug_fn = augmentations[aug_index % len(augmentations)]
    return aug_fn(image)


def _random_crop_resize(img: Image.Image) -> Image.Image:
    """Rastgele kırp ve orijinal boyuta geri getir."""
    w, h = img.size
    crop_ratio = random.uniform(0.75, 0.95)
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def augment_class(species: str, disease_class: str, min_count: int):
    """Belirli bir sınıf için görüntü sayısını min_count'a tamamla."""
    config = SPECIES_CONFIG[species]
    class_dir = os.path.join(config["raw_data_dir"], disease_class)

    if not os.path.exists(class_dir):
        print(f"  ❌ Dizin yok: {class_dir}")
        return 0

    existing = [
        f for f in os.listdir(class_dir)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
        and not f.startswith("aug_")  # Daha önce augmented olanları sayma
    ]

    current_total = len([
        f for f in os.listdir(class_dir)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    ])

    if current_total >= min_count:
        return 0

    if len(existing) == 0:
        print(f"  ❌ Orijinal görüntü yok: {disease_class}")
        return 0

    needed = min_count - current_total
    created = 0

    print(f"  🔄 {disease_class}: {current_total} → {min_count} "
          f"(+{needed} augmented)")

    aug_index = 0
    while created < needed:
        # Rastgele orijinal seç
        source_file = random.choice(existing)
        source_path = os.path.join(class_dir, source_file)

        try:
            img = Image.open(source_path).convert("RGB")
            aug_img = augment_image(img, aug_index)

            # Yeni dosya adı
            stem = Path(source_file).stem
            ext = Path(source_file).suffix
            new_name = f"aug_{aug_index:04d}_{stem}{ext}"
            new_path = os.path.join(class_dir, new_name)

            if not os.path.exists(new_path):
                aug_img.save(new_path, quality=95)
                created += 1

            aug_index += 1

        except Exception as e:
            print(f"    ⚠️  Augmentation hatası: {source_file} — {e}")
            aug_index += 1
            continue

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Veri çoğaltma (augmentation) aracı"
    )
    parser.add_argument(
        "--species", type=str, required=True, choices=SUPPORTED_SPECIES,
    )
    parser.add_argument(
        "--min-per-class", type=int, default=50,
        help="Her sınıfta minimum görüntü sayısı (varsayılan: 50)",
    )
    parser.add_argument(
        "--target-class", type=str, default=None,
        help="Sadece belirli bir sınıf için çoğalt",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    config = SPECIES_CONFIG[args.species]
    print(f"\n{'='*60}")
    print(f"  🔄 VERİ ÇOĞALTMA — {config['display_name']}")
    print(f"{'='*60}")
    print(f"  Hedef: Her sınıfta en az {args.min_per_class} görüntü\n")

    classes = [args.target_class] if args.target_class else DISEASE_CLASSES
    total_created = 0

    for cls in classes:
        if args.dry_run:
            class_dir = os.path.join(config["raw_data_dir"], cls)
            if os.path.exists(class_dir):
                count = len([
                    f for f in os.listdir(class_dir)
                    if Path(f).suffix.lower() in VALID_EXTENSIONS
                ])
                needed = max(0, args.min_per_class - count)
                print(f"  {cls}: {count} mevcut, {needed} gerekli")
        else:
            created = augment_class(args.species, cls, args.min_per_class)
            total_created += created

    if not args.dry_run:
        print(f"\n  ✅ Toplam {total_created} augmented görüntü oluşturuldu\n")
    else:
        print(f"\n  ℹ️  Dry-run modu — değişiklik yapılmadı\n")


if __name__ == "__main__":
    main()
