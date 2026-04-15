#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ham dataset'i train/val/test split'lerine ayır (çok-türlü).

Kullanım:
    python rebuild_dataset.py --species chicken
    python rebuild_dataset.py --species goose --split 0.7 0.15 0.15
    python rebuild_dataset.py --species duck
"""

import argparse
import os
import sys
import shutil
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    get_species_config,
    check_dataset_exists,
    SUPPORTED_SPECIES,
    DISEASE_CLASSES,
)


def rebuild_dataset(species: str, split_ratios: tuple = (0.7, 0.15, 0.15),
                    seed: int = 42):
    """Dataset'i train/val/test olarak böl."""
    random.seed(seed)

    config = get_species_config(species)
    source_dir = config["raw_data_dir"]
    output_dir = config["split_data_dir"]

    print(f"\n{'='*60}")
    print(f"  DATASET REBUILD — {config['display_name']}")
    print(f"{'='*60}")
    print(f"  Kaynak : {source_dir}")
    print(f"  Çıktı  : {output_dir}")
    print(f"  Split  : Train {split_ratios[0]:.0%} / "
          f"Val {split_ratios[1]:.0%} / Test {split_ratios[2]:.0%}")

    if not os.path.exists(source_dir):
        print(f"\n  ❌ Kaynak dizin bulunamadı: {source_dir}")
        print(f"     Önce: python create_species_folders.py")
        sys.exit(1)

    # Mevcut çıktıyı temizle
    if os.path.exists(output_dir):
        response = input(f"\n  ⚠️ {output_dir} zaten mevcut. Silinsin mi? (e/h): ")
        if response.lower() == "e":
            shutil.rmtree(output_dir)
        else:
            print("  İptal edildi.")
            sys.exit(0)

    splits = ["train", "val", "test"]
    stats = {s: {} for s in splits}
    total_copied = 0

    for disease in DISEASE_CLASSES:
        disease_dir = os.path.join(source_dir, disease)
        if not os.path.exists(disease_dir):
            print(f"  ⚠️  Atlanıyor (dizin yok): {disease}")
            continue

        # Görüntü dosyalarını bul
        images = [
            f for f in os.listdir(disease_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if len(images) == 0:
            print(f"  ⚠️  Atlanıyor (görüntü yok): {disease}")
            continue

        # Karıştır
        random.shuffle(images)

        # Split indeksleri
        n = len(images)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        split_images = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name in splits:
            split_disease_dir = os.path.join(output_dir, split_name, disease)
            os.makedirs(split_disease_dir, exist_ok=True)

            for img_name in split_images[split_name]:
                src = os.path.join(disease_dir, img_name)
                dst = os.path.join(split_disease_dir, img_name)
                shutil.copy2(src, dst)
                total_copied += 1

            stats[split_name][disease] = len(split_images[split_name])

    # Rapor
    print(f"\n  {'Sınıf':<30} {'Train':>6} {'Val':>6} {'Test':>6} {'Toplam':>7}")
    print(f"  {'-'*62}")

    for disease in DISEASE_CLASSES:
        t = stats["train"].get(disease, 0)
        v = stats["val"].get(disease, 0)
        te = stats["test"].get(disease, 0)
        total = t + v + te
        if total > 0:
            print(f"  {disease:<30} {t:>6} {v:>6} {te:>6} {total:>7}")

    total_train = sum(stats["train"].values())
    total_val = sum(stats["val"].values())
    total_test = sum(stats["test"].values())

    print(f"  {'-'*62}")
    print(f"  {'TOPLAM':<30} {total_train:>6} {total_val:>6} "
          f"{total_test:>6} {total_copied:>7}")
    print(f"\n  ✅ Tamamlandı! Çıktı dizini: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset'i train/val/test olarak böl (çok-türlü)"
    )
    parser.add_argument(
        "--species", type=str, default="chicken", choices=SUPPORTED_SPECIES,
        help="Hayvan türü (varsayılan: chicken)",
    )
    parser.add_argument(
        "--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split oranları (varsayılan: 0.7 0.15 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Split oranları doğrulama
    total = sum(args.split)
    if abs(total - 1.0) > 0.01:
        print(f"❌ Split oranları toplamı 1.0 olmalı (şu an: {total})")
        sys.exit(1)

    rebuild_dataset(args.species, tuple(args.split), args.seed)


if __name__ == "__main__":
    main()
