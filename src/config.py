#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Çok-türlü (multi-species) konfigürasyon modülü.

Kullanım:
    from src.config import get_config, SPECIES_CONFIG

    config = get_config("vit_b16", species="goose")
    print(config["data_dir"])
    print(config["num_classes"])
"""

import os
from copy import deepcopy

# ─────────────────────────────────────────────
# Proje kök dizini
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# Hastalık sınıfları (tüm türler için ortak)
# ─────────────────────────────────────────────
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

NUM_CLASSES = len(DISEASE_CLASSES)

# ─────────────────────────────────────────────
# Tür bazlı konfigürasyon
# ─────────────────────────────────────────────
SPECIES_CONFIG = {
    "chicken": {
        "display_name": "Tavuk 🐔",
        "raw_data_dir": os.path.join(BASE_DIR, "final_dataset_10_classes"),
        "split_data_dir": os.path.join(BASE_DIR, "chicken_split_dataset"),
        "data_dir": os.path.join(BASE_DIR, "final_dataset_10_classes"),
        "output_prefix": "chicken",
        "model_dir": os.path.join(BASE_DIR, "models", "chicken"),
        "results_dir": os.path.join(BASE_DIR, "results", "chicken"),
    },
    "goose": {
        "display_name": "Kaz 🦢",
        "raw_data_dir": os.path.join(BASE_DIR, "goose_dataset_10_classes"),
        "split_data_dir": os.path.join(BASE_DIR, "goose_split_dataset"),
        "data_dir": os.path.join(BASE_DIR, "goose_dataset_10_classes"),
        "output_prefix": "goose",
        "model_dir": os.path.join(BASE_DIR, "models", "goose"),
        "results_dir": os.path.join(BASE_DIR, "results", "goose"),
    },
    "duck": {
        "display_name": "Ördek 🦆",
        "raw_data_dir": os.path.join(BASE_DIR, "duck_dataset_10_classes"),
        "split_data_dir": os.path.join(BASE_DIR, "duck_split_dataset"),
        "data_dir": os.path.join(BASE_DIR, "duck_dataset_10_classes"),
        "output_prefix": "duck",
        "model_dir": os.path.join(BASE_DIR, "models", "duck"),
        "results_dir": os.path.join(BASE_DIR, "results", "duck"),
    },
}

SUPPORTED_SPECIES = list(SPECIES_CONFIG.keys())

# ─────────────────────────────────────────────
# Ortak konfigürasyon (tüm türler için)
# ─────────────────────────────────────────────
COMMON_CONFIG = {
    "num_classes": NUM_CLASSES,
    "class_names": DISEASE_CLASSES,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "early_stopping_patience": 5,
    "seed": 42,
}

# ─────────────────────────────────────────────
# Model bazlı konfigürasyon
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "vit_b16": {
        "model_name": "vit_b16",
        "display_name": "Vision Transformer (ViT-B/16)",
        "image_size": 224,
        "batch_size": 16,
        "learning_rate": 3e-4,
        "epochs": 30,
        "pretrained_weights": "imagenet21k",
    },
    "resnet50": {
        "model_name": "resnet50",
        "display_name": "ResNet-50",
        "image_size": 224,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 50,
    },
    "efficientnet_b0": {
        "model_name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "image_size": 224,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 50,
    },
    "mobilenet_v2": {
        "model_name": "mobilenet_v2",
        "display_name": "MobileNetV2",
        "image_size": 224,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 40,
    },
    "resnext50": {
        "model_name": "resnext50",
        "display_name": "ResNeXt-50",
        "image_size": 224,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "epochs": 30,
    },
    "resnest50d": {
        "model_name": "resnest50d",
        "display_name": "ResNeSt-50d",
        "image_size": 224,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "epochs": 30,
    },
    "convnext_tiny": {
        "model_name": "convnext_tiny",
        "display_name": "ConvNeXt-Tiny",
        "image_size": 224,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 30,
    },
    "cvt_13": {
        "model_name": "cvt_13",
        "display_name": "CvT-13",
        "image_size": 224,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 30,
    },
}


def validate_species(species: str) -> str:
    """Tür adını doğrula ve normalize et."""
    species = species.lower().strip()
    if species not in SUPPORTED_SPECIES:
        raise ValueError(
            f"Desteklenmeyen tür: '{species}'. "
            f"Desteklenen türler: {SUPPORTED_SPECIES}"
        )
    return species


def get_species_config(species: str = "chicken") -> dict:
    """Belirli bir tür için konfigürasyonu döndür."""
    species = validate_species(species)
    config = deepcopy(COMMON_CONFIG)
    config.update(deepcopy(SPECIES_CONFIG[species]))
    config["species"] = species
    return config


def get_config(model_name: str = "vit_b16", species: str = "chicken") -> dict:
    """
    Model + tür birleşik konfigürasyonu döndür.

    Args:
        model_name: Model adı (vit_b16, resnet50, vb.)
        species: Tür adı (chicken, goose, duck)

    Returns:
        Birleşik konfigürasyon dict'i

    Örnek:
        config = get_config("vit_b16", "goose")
        print(config["data_dir"])      # .../goose_dataset_10_classes
        print(config["model_name"])    # vit_b16
        print(config["species"])       # goose
    """
    species = validate_species(species)

    # Temel config
    config = deepcopy(COMMON_CONFIG)

    # Tür config
    config.update(deepcopy(SPECIES_CONFIG[species]))

    # Model config (varsa)
    if model_name in MODEL_CONFIGS:
        model_conf = deepcopy(MODEL_CONFIGS[model_name])
        config.update(model_conf)
    else:
        config["model_name"] = model_name

    config["species"] = species

    # Model save path - gerçek konumlara göre
    model_paths = {
        "vit_b16": os.path.join(BASE_DIR, "vit_poultry_results", "final_model"),
        "resnext50": os.path.join(BASE_DIR, "resnext_poultry_results", "best_resnext.pth"),
        "resnest50d": os.path.join(BASE_DIR, "resnest_poultry_results", "best_resnest.pth"),
        "convnext_tiny": os.path.join(BASE_DIR, "convnext_poultry_results", "final_model"),
        "cvt_13": os.path.join(BASE_DIR, "cvt_poultry_results", "final_model"),
        "resnet50": os.path.join(BASE_DIR, "models", "chicken", "resnet50_best.pth"),
        "efficientnet_b0": os.path.join(BASE_DIR, "models", "chicken", "efficientnet_b0_best.pth"),
        "mobilenet_v2": os.path.join(BASE_DIR, "models", "chicken", "mobilenet_v2_best.pth"),
    }
    
    config["experiment_name"] = f"{species}_{model_name}"
    config["output_dir"] = os.path.join(
        BASE_DIR, f"{model_name}_poultry_results"
    )
    config["model_save_path"] = model_paths.get(model_name, os.path.join(
        config["model_dir"], f"{model_name}_best.pth"
    ))

    return config


def get_model_path(model_name: str, species: str = "chicken") -> str:
    """Kaydedilmiş model dosya yolunu döndür."""
    config = get_config(model_name, species)
    return config["model_save_path"]


def check_dataset_exists(species: str) -> dict:
    """Belirli bir tür için dataset durumunu kontrol et."""
    species = validate_species(species)
    sp_config = SPECIES_CONFIG[species]

    report = {
        "species": species,
        "raw_data_dir": sp_config["raw_data_dir"],
        "exists": os.path.exists(sp_config["raw_data_dir"]),
        "classes": {},
        "total_images": 0,
        "ready_for_training": False,
    }

    if report["exists"]:
        for cls in DISEASE_CLASSES:
            cls_path = os.path.join(sp_config["raw_data_dir"], cls)
            if os.path.exists(cls_path):
                images = [
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                ]
                count = len(images)
            else:
                count = 0
            report["classes"][cls] = count
            report["total_images"] += count

        # En az her sınıfta 10 görüntü varsa eğitime hazır say
        report["ready_for_training"] = all(
            c >= 10 for c in report["classes"].values()
        )

    return report


# ─────────────────────────────────────────────
# Geriye uyumluluk fonksiyonları
# ─────────────────────────────────────────────
def get_device():
    """Cihaz döndür (geriye uyumlu)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_windows_encoding():
    """Windows encoding düzelt (geriye uyumlu)."""
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# ─────────────────────────────────────────────
# Modül doğrudan çalıştırılırsa durum raporu
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ÇOK-TÜRLÜ KONFİGÜRASYON RAPORU")
    print("=" * 60)

    for sp in SUPPORTED_SPECIES:
        report = check_dataset_exists(sp)
        status = "✅ HAZIR" if report["ready_for_training"] else "⏳ VERİ BEKLENİYOR"
        print(f"\n  {SPECIES_CONFIG[sp]['display_name']}")
        print(f"    Dizin  : {report['raw_data_dir']}")
        print(f"    Mevcut : {'Evet' if report['exists'] else 'Hayır'}")
        print(f"    Toplam : {report['total_images']} görüntü")
        print(f"    Durum  : {status}")

        if report["exists"] and report["total_images"] > 0:
            for cls, count in report["classes"].items():
                bar = "█" * min(count // 5, 20)
                print(f"      {cls:<30} {count:>5}  {bar}")

    print()
