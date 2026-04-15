"""
Dataset yukleme ve augmentation.
ImageFolder tabanli, corrupt gorsel korumali.
"""

import os
import logging
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile

# Truncated gorselleri yukle (bazi corrupt gorseller icin)
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class SafeImageFolder(datasets.ImageFolder):
    """Corrupt gorselleri atlayan ImageFolder."""

    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except Exception as e:
                path, _ = self.samples[index]
                logger.warning(f"Corrupt gorsel atlandi: {path} - {e}")
                index = (index + 1) % len(self.samples)


def get_transforms(image_size, split="train"):
    """Split'e gore transform pipeline dondurur."""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # val ve test icin deterministic
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def compute_class_weights(dataset):
    """Sinif agirliklarini hesaplar (dengesizlik icin)."""
    targets = [s[1] for s in dataset.samples]
    counter = Counter(targets)
    total = len(targets)
    num_classes = len(counter)

    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)
        w = total / (num_classes * count)
        weights.append(w)

    return torch.FloatTensor(weights)


def get_class_distribution(dataset):
    """Sinif dagilimini dondurur."""
    targets = [s[1] for s in dataset.samples]
    counter = Counter(targets)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return {idx_to_class[i]: counter[i] for i in sorted(counter.keys())}


def create_dataloaders(cfg):
    """Train, val, test DataLoader'larini olusturur."""
    data_dir = Path(cfg["data_dir"])
    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    loaders = {}
    class_names = None
    class_weights = None

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split klasoru bulunamadi: {split_dir}")
            continue

        transform = get_transforms(image_size, split)
        dataset = SafeImageFolder(str(split_dir), transform=transform)

        if class_names is None:
            class_names = dataset.classes
            cfg["num_classes"] = len(class_names)
            logger.info(f"Siniflar ({len(class_names)}): {class_names}")

        dist = get_class_distribution(dataset)
        logger.info(f"{split} dagilimi: {dist}")

        if split == "train" and cfg.get("use_class_weights", True):
            class_weights = compute_class_weights(dataset)
            logger.info(f"Sinif agirliklari: {class_weights.tolist()}")

        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train" and len(dataset) > batch_size),
        )

    return loaders, class_names, class_weights
