#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eğitilmiş modeli değerlendir (çok-türlü).

Kullanım:
    python evaluate_model.py --model vit_b16 --species chicken
    python evaluate_model.py --model vit_b16 --species goose
"""

import argparse
import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import get_config, SUPPORTED_SPECIES, DISEASE_CLASSES
from train_model import build_model


def plot_confusion_matrix(cm, class_names, save_path):
    """Confusion matrix görselleştir ve kaydet."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Tahmin", fontsize=12)
        ax.set_ylabel("Gerçek", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix kaydedildi: {save_path}")
    except ImportError:
        print("  ⚠️ matplotlib/seaborn yüklü değil, görsel oluşturulamadı.")


def main():
    parser = argparse.ArgumentParser(
        description="Model değerlendirme (çok-türlü)"
    )
    parser.add_argument(
        "--model", type=str, default="vit_b16",
        choices=["vit_b16", "resnext50", "resnest50d", "convnext_tiny", "cvt_13", "resnet50", "efficientnet_b0", "mobilenet_v2"],
    )
    parser.add_argument(
        "--species", type=str, default="chicken",
        choices=SUPPORTED_SPECIES,
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    config = get_config(args.model, args.species)

    # Model yolu
    model_path = args.model_path or config["model_save_path"]
    
    # HuggingFace formatı (klasör) veya PyTorch formatı (dosya) kontrolü
    is_huggingface_format = os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))
    is_pytorch_format = os.path.isfile(model_path)
    
    if not is_huggingface_format and not is_pytorch_format:
        print(f"\n❌ Model dosyası bulunamadı: {model_path}")
        print(f"   Önce eğitim yapın:")
        print(f"     python train_model.py --model {args.model} --species {args.species}")
        sys.exit(1)

    # Test veri dizini
    test_dir = args.data_dir
    if test_dir is None:
        split_dir = config.get("split_data_dir", config["data_dir"])
        test_dir = os.path.join(split_dir, "test")
        if not os.path.exists(test_dir):
            test_dir = os.path.join(split_dir, "val")
        if not os.path.exists(test_dir):
            test_dir = config["data_dir"]

    print(f"\n{'='*60}")
    print(f"  MODEL DEĞERLENDİRME")
    print(f"{'='*60}")
    print(f"  Tür     : {config.get('display_name', args.species)}")
    print(f"  Model   : {args.model}")
    print(f"  Ağırlık : {model_path}")
    print(f"  Test Dir: {test_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    test_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4,
    )

    print(f"  Test    : {len(test_dataset)} görüntü")
    class_names = test_dataset.classes
    print(f"  Sınıflar: {class_names}")

    # Model yükle
    # HuggingFace formatı (klasör) kontrolü
    if is_huggingface_format:
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(model_path)
    else:
        model = build_model(args.model, config["num_classes"], pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        
        # HuggingFace modelleri için farklı yükleme
        if args.model in ["convnext_tiny", "cvt_13"]:
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
        elif args.model == "resnest50d":
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
        else:
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
    
    model = model.to(device)
    model.eval()

    # Tahmin
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # HuggingFace modelleri ModelOutput döner
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrikler
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'='*60}")
    print(f"  SONUÇLAR")
    print(f"{'='*60}")
    print(f"  Accuracy     : {accuracy:.4f} ({accuracy:.2%})")
    print(f"  F1 (macro)   : {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Sonuçları kaydet
    results = {
        "species": args.species,
        "model": args.model,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {results_path}")


if __name__ == "__main__":
    main()
