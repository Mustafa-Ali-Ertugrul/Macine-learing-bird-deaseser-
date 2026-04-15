"""
Kaz ve ördek için boş dataset klasör yapısını oluşturur.
Mevcut chicken klasörüne dokunmaz.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISEASE_CLASSES = [
    "Avian_Influenza",
    "Coccidiosis",
    "Fowl_Pox",
    "Healthy",
    "Histomoniasis",
    "Infectious_Bronchitis",
    "Infectious_Bursal_Disease",
    "Mareks_Disease",
    "Newcastle_Disease",
    "Salmonella",
]

SPECIES_DATASETS = {
    "goose": "goose_dataset_10_classes",
    "duck": "duck_dataset_10_classes",
}

SPLITS = ["train", "val", "test"]


def create_folders():
    created = []
    skipped = []

    for species, dataset_dir_name in SPECIES_DATASETS.items():
        dataset_path = os.path.join(BASE_DIR, dataset_dir_name)

        for disease in DISEASE_CLASSES:
            # Üst düzey (raw) klasör
            raw_path = os.path.join(dataset_path, disease)
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)
                created.append(raw_path)
            else:
                skipped.append(raw_path)

        # Split klasörleri (rebuild_dataset.py çıktısı için)
        split_base = os.path.join(BASE_DIR, f"{species}_split_dataset")
        for split in SPLITS:
            for disease in DISEASE_CLASSES:
                split_path = os.path.join(split_base, split, disease)
                if not os.path.exists(split_path):
                    os.makedirs(split_path)
                    created.append(split_path)
                else:
                    skipped.append(split_path)

    # Placeholder dosyaları (Git'in boş klasörleri izlemesi için)
    for species, dataset_dir_name in SPECIES_DATASETS.items():
        dataset_path = os.path.join(BASE_DIR, dataset_dir_name)
        readme_path = os.path.join(dataset_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"# {species.capitalize()} Dataset\n\n")
                f.write("Bu klasör henüz boştur.\n")
                f.write("Her alt klasöre ilgili hastalık görüntülerini ekleyin.\n\n")
                f.write("## Hastalık Sınıfları\n\n")
                for d in DISEASE_CLASSES:
                    f.write(f"- `{d}/`\n")
            created.append(readme_path)

    print(f"\n{'='*60}")
    print(f"  KLASÖR YAPISI RAPORU")
    print(f"{'='*60}")
    print(f"  Oluşturulan : {len(created)}")
    print(f"  Zaten mevcut: {len(skipped)}")
    print(f"{'='*60}\n")

    for path in created[:20]:  # İlk 20'yi göster
        print(f"  [+] {os.path.relpath(path, BASE_DIR)}")
    if len(created) > 20:
        print(f"  ... ve {len(created) - 20} klasör daha")


if __name__ == "__main__":
    create_folders()
