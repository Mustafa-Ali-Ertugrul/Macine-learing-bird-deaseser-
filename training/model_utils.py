"""
Model olusturma ve yonetimi.
EfficientNet-B0, ResNet50, MobileNet V2/V3, ResNet18 ve ViT-B/16 destegi.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def create_model(cfg):
    """Pretrained model olusturur ve son katmani num_classes'a uyarlar."""
    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]
    pretrained = cfg["pretrained"]
    freeze_backbone = cfg["freeze_backbone"]

    logger.info(f"Model: {model_name} | Sinif: {num_classes} | Pretrained: {pretrained}")

    weights = "DEFAULT" if pretrained else None

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "resnet18":
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name in ("vit_b16", "vit_b_16"):
        model = models.vit_b_16(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "heads" not in name:
                    param.requires_grad = False
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    # Kismi unfreeze: backbone donuk ama son N blok acik
    unfreeze_n = cfg.get("unfreeze_last_n", 0)
    if freeze_backbone and unfreeze_n > 0:
        if model_name == "efficientnet_b0":
            blocks = list(model.features.children())
            for block in blocks[-unfreeze_n:]:
                for param in block.parameters():
                    param.requires_grad = True
            logger.info(f"EfficientNet: son {unfreeze_n}/{len(blocks)} blok acildi")

        elif model_name in ("resnet50", "resnet18"):
            layers = [model.layer1, model.layer2, model.layer3, model.layer4]
            for layer in layers[-unfreeze_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"ResNet: son {unfreeze_n}/4 layer acildi")

        elif model_name in ("mobilenet_v2", "mobilenet_v3"):
            blocks = list(model.features.children())
            for block in blocks[-unfreeze_n:]:
                for param in block.parameters():
                    param.requires_grad = True
            logger.info(f"MobileNet: son {unfreeze_n}/{len(blocks)} blok acildi")

        elif model_name in ("vit_b16", "vit_b_16"):
            blocks = list(model.encoder.layers.children())
            for block in blocks[-unfreeze_n:]:
                for param in block.parameters():
                    param.requires_grad = True
            logger.info(f"ViT: son {unfreeze_n}/{len(blocks)} encoder blogu acildi")

    # Egitilecek parametre sayisi
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parametreler: {trainable:,} / {total:,} ({trainable/total*100:.1f}% egitilecek)")

    return model


def get_optimizer(model, cfg):
    """Optimizer olusturur."""
    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]

    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd,
        )
    elif cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd,
        )
    elif cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=0.9, weight_decay=wd,
        )
    else:
        raise ValueError(f"Desteklenmeyen optimizer: {cfg['optimizer']}")


def get_scheduler(optimizer, cfg):
    """Learning rate scheduler olusturur."""
    if cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["t_max"], eta_min=1e-6,
        )
    elif cfg["scheduler"] == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"],
        )
    elif cfg["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True,
        )
    else:
        return None
