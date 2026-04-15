#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Çok-türlü hastalık sınıflandırma eğitim pipeline'ı.

Kullanım:
    # Tavuk (varsayılan - geriye uyumlu)
    python train_model.py --model vit_b16

    # Kaz
    python train_model.py --model vit_b16 --species goose

    # Ördek
    python train_model.py --model vit_b16 --species duck

    # YAML config ile
    python train_model.py --config training_config_goose.yaml
"""

import argparse
import os
import sys
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# Proje modülleri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    get_config,
    validate_species,
    check_dataset_exists,
    SUPPORTED_SPECIES,
    DISEASE_CLASSES,
)


def set_seed(seed: int = 42):
    """Tekrarlanabilirlik için tüm random seed'leri ayarla."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_transforms(image_size: int = 224):
    """Eğitim ve doğrulama için veri dönüşümleri."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Model oluştur."""
    import torchvision.models as models

    if model_name == "vit_b16":
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name == "resnet50":
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        if pretrained:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif model_name == "mobilenet_v2":
        if pretrained:
            model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V2
            )
        else:
            model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif model_name == "resnext50":
        if pretrained:
            model = models.resnext50_32x4d(
                weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
            )
        else:
            model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "convnext_tiny":
        from transformers import ConvNextForImageClassification
        if pretrained:
            model = ConvNextForImageClassification.from_pretrained(
                "facebook/convnext-tiny-224",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            model = ConvNextForImageClassification.from_pretrained(
                "facebook/convnext-tiny-224",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

    elif model_name == "resnest50d":
        import timm
        model = timm.create_model(
            "resnest50d",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif model_name == "cvt_13":
        from transformers import CvtForImageClassification
        if pretrained:
            model = CvtForImageClassification.from_pretrained(
                "microsoft/cvt-13",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            model = CvtForImageClassification.from_pretrained(
                "microsoft/cvt-13",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Tek epoch eğitim."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Doğrulama."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(
        description="Çok-türlü kümes hayvanı hastalık sınıflandırma eğitimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python train_model.py --model vit_b16 --species chicken
  python train_model.py --model resnext50 --species chicken
  python train_model.py --model convnext_tiny --species chicken
  python train_model.py --model resnest50d --species chicken
  python train_model.py --model cvt_13 --species chicken
  python train_model.py --model resnet50 --species goose
  python train_model.py --model efficientnet_b0 --species duck
  python train_model.py --model vit_b16  (varsayılan: chicken)
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_b16",
        choices=["vit_b16", "resnext50", "resnest50d", "convnext_tiny", "cvt_13", "resnet50", "efficientnet_b0", "mobilenet_v2"],
        help="Model mimarisi (varsayılan: vit_b16)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="chicken",
        choices=SUPPORTED_SPECIES,
        help="Hayvan türü (varsayılan: chicken)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=None, help="Öğrenme oranı")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Veri dizini (belirtilmezse config'den alınır)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint'tan devam")
    parser.add_argument("--no-pretrained", action="store_true", help="Sıfırdan eğit")

    args = parser.parse_args()

    # ── Konfigürasyon ──
    config = get_config(args.model, args.species)

    # CLI argümanları ile override
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.data_dir:
        config["data_dir"] = args.data_dir

    # ── Dataset kontrol ──
    dataset_report = check_dataset_exists(args.species)
    if not dataset_report["exists"]:
        print(f"\n❌ HATA: {args.species} için dataset dizini bulunamadı!")
        print(f"   Beklenen dizin: {dataset_report['raw_data_dir']}")
        print(f"\n   Önce klasör yapısını oluşturun:")
        print(f"     python create_species_folders.py")
        print(f"\n   Sonra görüntüleri ekleyin.")
        sys.exit(1)

    if not dataset_report["ready_for_training"]:
        print(f"\n⚠️  UYARI: {args.species} dataset'i eğitim için yeterli değil!")
        print(f"   Toplam görüntü: {dataset_report['total_images']}")
        print(f"\n   Sınıf bazlı dağılım:")
        for cls, count in dataset_report["classes"].items():
            status = "✅" if count >= 10 else "❌"
            print(f"     {status} {cls}: {count}")
        print(f"\n   Her sınıfta en az 10 görüntü gereklidir.")

        response = input("\n   Yine de devam etmek istiyor musunuz? (e/h): ")
        if response.lower() != "e":
            sys.exit(0)

    # ── Başlık ──
    species_display = config.get("display_name", args.species)
    print(f"\n{'='*60}")
    print(f"  🐦 KÜMES HAYVANI HASTALIK SINIFLANDIRMA EĞİTİMİ")
    print(f"{'='*60}")
    print(f"  Tür        : {species_display}")
    print(f"  Model      : {config.get('display_name', args.model)}")
    print(f"  Veri Dizini: {config['data_dir']}")
    print(f"  Epoch      : {config['epochs']}")
    print(f"  Batch      : {config['batch_size']}")
    print(f"  LR         : {config['learning_rate']}")
    print(f"  Sınıf Sayısı: {config['num_classes']}")
    print(f"{'='*60}\n")

    # ── Seed ──
    set_seed(config["seed"])

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # ── Data Loaders ──
    train_transform, val_transform = get_data_transforms(config["image_size"])

    # Split dizin yapısını kontrol et
    split_dir = config.get("split_data_dir", config["data_dir"])
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")

    # Eğer split yapılmamışsa doğrudan data_dir kullan
    if not os.path.exists(train_dir):
        print(f"\n⚠️  Split dizini bulunamadı: {train_dir}")
        print(f"   Doğrudan data_dir kullanılıyor: {config['data_dir']}")
        print(f"   Daha iyi sonuç için önce split yapın:")
        print(f"     python rebuild_dataset.py --species {args.species}\n")
        train_dir = config["data_dir"]
        val_dir = config["data_dir"]

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  Train: {len(train_dataset)} görüntü")
    print(f"  Val  : {len(val_dataset)} görüntü")
    print(f"  Sınıflar: {train_dataset.classes}\n")

    # ── Model ──
    model = build_model(
        args.model,
        config["num_classes"],
        pretrained=not args.no_pretrained,
    )
    model = model.to(device)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"  Checkpoint yüklendi: {args.resume} (epoch {start_epoch})")

    # ── Optimizer & Scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # ── Output dizinleri ──
    output_dir = config["output_dir"]
    model_dir = os.path.dirname(config["model_save_path"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── Eğitim döngüsü ──
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10} | {'Durum':>8}")
    print(f"  {'-'*75}")

    start_time = time.time()

    for epoch in range(start_epoch, config["epochs"]):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Best model
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            status = "✅ BEST"

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "species": args.species,
                    "model_name": args.model,
                    "class_names": DISEASE_CLASSES,
                    "config": config,
                },
                config["model_save_path"],
            )
        else:
            patience_counter += 1

        print(
            f"  {epoch+1:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
            f"{val_loss:>8.4f} | {val_acc:>6.2%} | {current_lr:>10.6f} | {status}"
        )

        # Early stopping
        if patience_counter >= config["early_stopping_patience"]:
            print(f"\n  ⏹ Early stopping: {config['early_stopping_patience']} "
                  f"epoch boyunca iyileşme yok.")
            break

    elapsed = time.time() - start_time

    # ── Özet ──
    print(f"\n{'='*60}")
    print(f"  EĞİTİM TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Tür          : {species_display}")
    print(f"  Model        : {args.model}")
    print(f"  En İyi Val Acc: {best_val_acc:.2%}")
    print(f"  Süre         : {elapsed/60:.1f} dakika")
    print(f"  Model Kaydı  : {config['model_save_path']}")
    print(f"{'='*60}\n")

    # ── History kaydet ──
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "species": args.species,
                "model": args.model,
                "best_val_acc": best_val_acc,
                "elapsed_seconds": elapsed,
                "history": history,
                "config": {
                    k: str(v) if not isinstance(v, (int, float, str, list, bool))
                    else v
                    for k, v in config.items()
                },
            },
            f,
            indent=2,
        )
    print(f"  Training history: {history_path}")


if __name__ == "__main__":
    main()
