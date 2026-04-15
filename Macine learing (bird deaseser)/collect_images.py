#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Görüntü toplama yardımcı scripti.

Bu script:
1. Kaggle'dan dataset indirir (API key gerekli)
2. Mevcut görüntüleri doğru klasörlere kopyalar
3. Görüntüleri doğrular ve bozuk olanları temizler
4. Her sınıf için durum raporu verir

Kullanım:
    python collect_images.py --species goose --source-dir /path/to/raw/images
    python collect_images.py --species duck --source-dir /path/to/raw/images
    python collect_images.py --species goose --from-kaggle username/dataset-name
    python collect_images.py --validate-only --species goose
    python collect_images.py --status
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    SPECIES_CONFIG,
    SUPPORTED_SPECIES,
    DISEASE_CLASSES,
    check_dataset_exists,
)

# ─────────────────────────────────────
# Desteklenen görüntü uzantıları
# ─────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def validate_image(image_path: str) -> bool:
    """Görüntü dosyasının geçerli olup olmadığını kontrol et."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # verify() sonrası tekrar aç (verify dosyayı kapatır)
        with Image.open(image_path) as img:
            img.load()
            # Minimum boyut kontrolü
            if img.size[0] < 32 or img.size[1] < 32:
                return False
        return True
    except Exception:
        return False


def clean_and_validate_dataset(species: str):
    """
    Dataset'teki tüm görüntüleri doğrula.
    Bozuk olanları ayrı klasöre taşı.
    """
    config = SPECIES_CONFIG[species]
    data_dir = config["raw_data_dir"]

    if not os.path.exists(data_dir):
        print(f"❌ Dizin bulunamadı: {data_dir}")
        return

    corrupt_dir = os.path.join(data_dir, "_corrupt_images")

    total_valid = 0
    total_corrupt = 0
    stats = {}

    print(f"\n{'='*60}")
    print(f"  🔍 GÖRÜNTÜ DOĞRULAMA — {config['display_name']}")
    print(f"{'='*60}")
    print(f"  Dizin: {data_dir}\n")

    for disease in DISEASE_CLASSES:
        disease_dir = os.path.join(data_dir, disease)
        if not os.path.exists(disease_dir):
            stats[disease] = {"valid": 0, "corrupt": 0}
            continue

        valid_count = 0
        corrupt_count = 0

        files = [
            f for f in os.listdir(disease_dir)
            if Path(f).suffix.lower() in VALID_EXTENSIONS
        ]

        for filename in files:
            filepath = os.path.join(disease_dir, filename)

            if validate_image(filepath):
                valid_count += 1
            else:
                corrupt_count += 1
                # Bozuk dosyayı taşı
                corrupt_disease_dir = os.path.join(corrupt_dir, disease)
                os.makedirs(corrupt_disease_dir, exist_ok=True)
                shutil.move(filepath, os.path.join(corrupt_disease_dir, filename))

        stats[disease] = {"valid": valid_count, "corrupt": corrupt_count}
        total_valid += valid_count
        total_corrupt += corrupt_count

    # Rapor
    print(f"  {'Sınıf':<30} {'Geçerli':>8} {'Bozuk':>8} {'Durum':>8}")
    print(f"  {'-'*58}")

    for disease in DISEASE_CLASSES:
        s = stats[disease]
        status = "✅" if s["valid"] >= 10 else ("⚠️" if s["valid"] > 0 else "❌")
        print(
            f"  {disease:<30} {s['valid']:>8} {s['corrupt']:>8} {status:>8}"
        )

    print(f"  {'-'*58}")
    print(f"  {'TOPLAM':<30} {total_valid:>8} {total_corrupt:>8}")

    if total_corrupt > 0:
        print(f"\n  ⚠️  {total_corrupt} bozuk görüntü taşındı: {corrupt_dir}")

    print()
    return stats


def copy_from_source(species: str, source_dir: str, mapping: dict = None):
    """
    Kaynak dizinden hedef dataset dizinine görüntüleri kopyala.

    Args:
        species: Hedef tür
        source_dir: Kaynak dizin
        mapping: Kaynak klasör adı → hedef hastalık sınıfı eşlemesi
                 Örnek: {"healthy_goose": "Healthy", "flu_goose": "Avian_Influenza"}
                 None ise kaynak klasör adları doğrudan kullanılır
    """
    config = SPECIES_CONFIG[species]
    target_dir = config["raw_data_dir"]

    if not os.path.exists(source_dir):
        print(f"❌ Kaynak dizin bulunamadı: {source_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  📁 GÖRÜNTÜ KOPYALAMA — {config['display_name']}")
    print(f"{'='*60}")
    print(f"  Kaynak: {source_dir}")
    print(f"  Hedef : {target_dir}\n")

    copied = 0
    skipped = 0
    errors = 0

    # Kaynak dizindeki alt klasörleri tara
    if os.path.isdir(source_dir):
        source_folders = [
            d for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ]

        if not source_folders:
            # Düz dizin — tüm görüntüleri kullanıcıya sor
            print("  ⚠️  Alt klasör bulunamadı. Düz dizin yapısı algılandı.")
            print("  Görüntüler hangi hastalık sınıfına ait?")
            for i, cls in enumerate(DISEASE_CLASSES, 1):
                print(f"    {i:>2}. {cls}")

            choice = input("\n  Sınıf numarasını girin (1-10): ").strip()
            try:
                target_class = DISEASE_CLASSES[int(choice) - 1]
            except (ValueError, IndexError):
                print("  ❌ Geçersiz seçim.")
                return

            target_class_dir = os.path.join(target_dir, target_class)
            os.makedirs(target_class_dir, exist_ok=True)

            for f in os.listdir(source_dir):
                if Path(f).suffix.lower() in VALID_EXTENSIONS:
                    src = os.path.join(source_dir, f)
                    dst = os.path.join(target_class_dir, f)
                    if not os.path.exists(dst):
                        if validate_image(src):
                            shutil.copy2(src, dst)
                            copied += 1
                        else:
                            errors += 1
                    else:
                        skipped += 1
        else:
            # Alt klasörlü yapı
            for folder in source_folders:
                # Eşleme
                if mapping and folder in mapping:
                    target_class = mapping[folder]
                elif folder in DISEASE_CLASSES:
                    target_class = folder
                else:
                    # Fuzzy match dene
                    matched = _fuzzy_match_class(folder)
                    if matched:
                        target_class = matched
                        print(f"  🔄 '{folder}' → '{target_class}' (otomatik eşleme)")
                    else:
                        print(f"  ⚠️  Atlanıyor (eşleşme yok): {folder}")
                        continue

                source_class_dir = os.path.join(source_dir, folder)
                target_class_dir = os.path.join(target_dir, target_class)
                os.makedirs(target_class_dir, exist_ok=True)

                for f in os.listdir(source_class_dir):
                    if Path(f).suffix.lower() in VALID_EXTENSIONS:
                        src = os.path.join(source_class_dir, f)
                        # Çakışma önleme — tür prefix ekle
                        new_name = f"{species}_{f}" if not f.startswith(species) else f
                        dst = os.path.join(target_class_dir, new_name)

                        if not os.path.exists(dst):
                            if validate_image(src):
                                shutil.copy2(src, dst)
                                copied += 1
                            else:
                                errors += 1
                        else:
                            skipped += 1

    print(f"\n  ✅ Kopyalanan : {copied}")
    print(f"  ⏭️  Atlanan    : {skipped} (zaten mevcut)")
    print(f"  ❌ Hatalı     : {errors} (bozuk görüntü)")
    print()


def _fuzzy_match_class(folder_name: str) -> str:
    """Klasör adını hastalık sınıfıyla eşleştirmeye çalış."""
    folder_lower = folder_name.lower().replace(" ", "_").replace("-", "_")

    # Doğrudan eşleme sözlüğü
    aliases = {
        "avian_flu": "Avian_Influenza",
        "bird_flu": "Avian_Influenza",
        "flu": "Avian_Influenza",
        "influenza": "Avian_Influenza",
        "ai": "Avian_Influenza",
        "cocci": "Coccidiosis",
        "coccidiose": "Coccidiosis",
        "fowl_pox": "Fowl_Pox",
        "fowlpox": "Fowl_Pox",
        "pox": "Fowl_Pox",
        "healthy": "Healthy",
        "normal": "Healthy",
        "histo": "Histomoniasis",
        "blackhead": "Histomoniasis",
        "bronchitis": "Infectious_Bronchitis",
        "ib": "Infectious_Bronchitis",
        "infectious_bronchitis": "Infectious_Bronchitis",
        "ibd": "Infectious_Bursal_Disease",
        "gumboro": "Infectious_Bursal_Disease",
        "bursal": "Infectious_Bursal_Disease",
        "mareks": "Mareks_Disease",
        "marek": "Mareks_Disease",
        "newcastle": "Newcastle_Disease",
        "nd": "Newcastle_Disease",
        "salmonella": "Salmonella",
        "salmonellosis": "Salmonella",
    }

    for alias, cls in aliases.items():
        if alias in folder_lower:
            return cls

    # Kısmi eşleme
    for cls in DISEASE_CLASSES:
        if cls.lower() in folder_lower or folder_lower in cls.lower():
            return cls

    return None


def download_from_kaggle(dataset_slug: str, download_dir: str):
    """Kaggle'dan dataset indir."""
    try:
        import kaggle
        print(f"\n  📥 Kaggle'dan indiriliyor: {dataset_slug}")
        print(f"     Hedef: {download_dir}")

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_slug,
            path=download_dir,
            unzip=True,
        )
        print(f"  ✅ İndirme tamamlandı!")
        return True

    except ImportError:
        print("  ❌ kaggle paketi yüklü değil: pip install kaggle")
        print("  📋 Manuel adımlar:")
        print(f"     1. https://kaggle.com/datasets/{dataset_slug} adresine gidin")
        print(f"     2. 'Download' butonuna tıklayın")
        print(f"     3. ZIP dosyasını {download_dir} dizinine açın")
        return False
    except Exception as e:
        print(f"  ❌ Kaggle hatası: {e}")
        print("  📋 ~/.kaggle/kaggle.json dosyasını kontrol edin")
        return False


def generate_status_report():
    """Tüm türler için durum raporu oluştur."""
    print(f"\n{'='*70}")
    print(f"  📊 GENEL DURUM RAPORU")
    print(f"{'='*70}")

    for species in SUPPORTED_SPECIES:
        report = check_dataset_exists(species)
        config = SPECIES_CONFIG[species]
        display = config["display_name"]

        ready = report["ready_for_training"]
        total = report["total_images"]

        status_icon = "✅" if ready else ("🟡" if total > 0 else "❌")

        print(f"\n  {status_icon} {display}")
        print(f"     Dizin     : {report['raw_data_dir']}")
        print(f"     Toplam    : {total} görüntü")
        print(f"     Eğitime   : {'HAZIR' if ready else 'HAZIR DEĞİL'}")

        if report["exists"]:
            empty_classes = [
                cls for cls, count in report["classes"].items() if count == 0
            ]
            low_classes = [
                cls for cls, count in report["classes"].items()
                if 0 < count < 10
            ]

            if empty_classes:
                print(f"     ❌ Boş     : {', '.join(empty_classes)}")
            if low_classes:
                print(f"     ⚠️  Az     : {', '.join(low_classes)}")

            # Sınıf dağılımı
            print(f"     {'─'*50}")
            for cls in DISEASE_CLASSES:
                count = report["classes"].get(cls, 0)
                bar = "█" * min(count // 2, 25)
                status = "✅" if count >= 10 else ("⚠️" if count > 0 else "  ")
                print(f"     {status} {cls:<28} {count:>5}  {bar}")

    # Sonraki adımlar
    print(f"\n{'='*70}")
    print(f"  📋 SONRAKİ ADIMLAR")
    print(f"{'='*70}")

    for species in SUPPORTED_SPECIES:
        report = check_dataset_exists(species)
        if not report["ready_for_training"]:
            display = SPECIES_CONFIG[species]["display_name"]
            print(f"\n  {display}:")

            if report["total_images"] == 0:
                print(f"    1. Görüntüleri toplayın")
                print(f"    2. python collect_images.py --species {species} --source-dir /path/to/images")
                print(f"    3. python collect_images.py --validate-only --species {species}")
                print(f"    4. python rebuild_dataset.py --species {species}")
                print(f"    5. python train_model.py --model vit_b16 --species {species}")
            else:
                empty = [c for c, n in report["classes"].items() if n < 10]
                print(f"    Eksik sınıflar: {', '.join(empty)}")
                print(f"    Her sınıfta en az 10 görüntü gerekli")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Görüntü toplama ve hazırlama aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Kaynak dizinden kopyala
  python collect_images.py --species goose --source-dir ./raw_goose_images

  # Kaggle'dan indir
  python collect_images.py --species goose --from-kaggle user/goose-disease-dataset

  # Sadece doğrula
  python collect_images.py --validate-only --species goose

  # Genel durum raporu
  python collect_images.py --status

  # Tüm türler için doğrulama
  python collect_images.py --validate-all
        """,
    )

    parser.add_argument(
        "--species", type=str, default="goose",
        choices=SUPPORTED_SPECIES,
    )
    parser.add_argument("--source-dir", type=str, help="Kaynak görüntü dizini")
    parser.add_argument("--from-kaggle", type=str, help="Kaggle dataset slug")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Sadece mevcut görüntüleri doğrula",
    )
    parser.add_argument(
        "--validate-all", action="store_true",
        help="Tüm türlerin görüntülerini doğrula",
    )
    parser.add_argument("--status", action="store_true", help="Durum raporu")
    parser.add_argument(
        "--mapping", type=str, default=None,
        help='Klasör eşleme JSON: \'{"src_folder": "Disease_Class"}\'',
    )

    args = parser.parse_args()

    if args.status:
        generate_status_report()
        return

    if args.validate_all:
        for sp in SUPPORTED_SPECIES:
            clean_and_validate_dataset(sp)
        return

    if args.validate_only:
        clean_and_validate_dataset(args.species)
        return

    if args.from_kaggle:
        download_dir = os.path.join("downloads", args.species)
        os.makedirs(download_dir, exist_ok=True)
        success = download_from_kaggle(args.from_kaggle, download_dir)
        if success:
            mapping = None
            if args.mapping:
                import json
                mapping = json.loads(args.mapping)
            copy_from_source(args.species, download_dir, mapping)
            clean_and_validate_dataset(args.species)

    elif args.source_dir:
        mapping = None
        if args.mapping:
            import json
            mapping = json.loads(args.mapping)
        copy_from_source(args.species, args.source_dir, mapping)
        clean_and_validate_dataset(args.species)

    else:
        print("❌ --source-dir veya --from-kaggle belirtmelisiniz.")
        print("   Yardım için: python collect_images.py --help")
        print("   Durum için : python collect_images.py --status")


if __name__ == "__main__":
    main()
