#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tek görüntü tahmini (çok-türlü).

Kullanım:
    python predict_single.py --image foto.jpg --species chicken
    python predict_single.py --image foto.jpg --species goose --model resnet50
"""

import argparse
import os
import sys

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import get_config, SUPPORTED_SPECIES, DISEASE_CLASSES
from train_model import build_model


def predict(image_path: str, model_name: str = "vit_b16",
            species: str = "chicken", model_path: str = None,
            top_k: int = 3):
    """
    Tek bir görüntü için hastalık tahmini yap.

    Returns:
        dict: {
            "species": str,
            "predictions": [{"class": str, "confidence": float}, ...],
            "top_prediction": str,
            "confidence": float,
        }
    """
    config = get_config(model_name, species)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model yükle
    _model_path = model_path or config["model_save_path"]
    if not os.path.exists(_model_path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {_model_path}\n"
            f"Önce eğitim yapın: python train_model.py "
            f"--model {model_name} --species {species}"
        )

    model = build_model(model_name, config["num_classes"], pretrained=False)
    
    # HuggingFace modelleri için farklı yükleme
    if model_name in ["convnext_tiny", "cvt_13"]:
        if os.path.isdir(_model_path):
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(_model_path)
        else:
            checkpoint = torch.load(_model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
    elif model_name == "resnest50d":
        checkpoint = torch.load(_model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
    else:
        checkpoint = torch.load(_model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
    
    model = model.to(device)
    model.eval()

    # Sınıf isimleri — checkpoint'tan veya config'den
    class_names = checkpoint.get("class_names", DISEASE_CLASSES)

    # Görüntü ön-işleme
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Tahmin
    with torch.no_grad():
        outputs = model(input_tensor)
        # HuggingFace modelleri ModelOutput döner
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

    predictions = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        predictions.append({
            "class": class_names[idx],
            "confidence": round(prob, 4),
        })

    result = {
        "species": species,
        "species_display": config.get("display_name", species),
        "model": model_name,
        "image": image_path,
        "predictions": predictions,
        "top_prediction": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Tek görüntü hastalık tahmini (çok-türlü)"
    )
    parser.add_argument("--image", type=str, required=True, help="Görüntü yolu")
    parser.add_argument(
        "--model", type=str, default="vit_b16",
        choices=["vit_b16", "resnet50", "efficientnet_b0", "mobilenet_v2"],
    )
    parser.add_argument(
        "--species", type=str, default="chicken", choices=SUPPORTED_SPECIES,
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Görüntü bulunamadı: {args.image}")
        sys.exit(1)

    result = predict(
        args.image, args.model, args.species, args.model_path, args.top_k
    )

    species_display = result["species_display"]
    print(f"\n{'='*50}")
    print(f"  🐦 TAHMİN SONUCU — {species_display}")
    print(f"{'='*50}")
    print(f"  Görüntü : {result['image']}")
    print(f"  Model   : {result['model']}")
    print(f"  Tür     : {species_display}")
    print(f"\n  Top-{args.top_k} Tahminler:")

    for i, pred in enumerate(result["predictions"], 1):
        bar_len = int(pred["confidence"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        emoji = "🔴" if pred["class"] != "Healthy" else "🟢"
        print(
            f"    {i}. {emoji} {pred['class']:<30} "
            f"{pred['confidence']:>6.2%}  {bar}"
        )

    print(f"\n  ➡️  Sonuç: {result['top_prediction']} "
          f"({result['confidence']:.2%} güven)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
