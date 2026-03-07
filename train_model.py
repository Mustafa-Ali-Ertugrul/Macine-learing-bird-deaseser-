#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal Training Script for Poultry Disease Classification

Supports all 5 model architectures:
  - vit_b16      : Vision Transformer (torchvision)
  - convnext_tiny: ConvNeXt-Tiny (HuggingFace)
  - resnext50    : ResNeXt-50 (torchvision)
  - resnest50    : ResNeSt-50d (timm)
  - cvt          : CvT-13 (HuggingFace)

Features:
  - Clean dataset (no data leakage) via clean_dataset_split/
  - FocalLoss with per-class weights for imbalanced data
  - WeightedRandomSampler for balanced mini-batches
  - Mixed precision training (fp16)
  - Gradient accumulation
  - Strong augmentation pipeline (on-the-fly, train only)

Usage:
    python train_model.py --model vit_b16
    python train_model.py --model convnext_tiny
    python train_model.py --model resnext50
    python train_model.py --model resnest50
    python train_model.py --model cvt
    python train_model.py --model all   # Train all models sequentially
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models

import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config, fix_windows_encoding, get_device, COMMON_CONFIG
from src.dataset_utils import (
    PoultryImageDataset, HuggingFaceDataset, prepare_datasets,
    print_dataset_info, create_label_mappings,
    create_weighted_sampler, get_class_weights_tensor, create_dataloaders
)
from src.training_utils import (
    TrainerBase, HuggingFaceTrainer, print_summary,
    plot_training_history, save_results, setup_optimizer_scheduler
)
from src.losses import FocalLoss

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Augmentation
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Strong augmentation pipeline for training.
    Applied on-the-fly -- only to training split.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # applied after ToTensor
        # Note: RandomErasing expects tensor, so we put ToTensor before it
    ])


def get_train_transforms_with_tensor(img_size: int = 224) -> transforms.Compose:
    """Train transforms including ToTensor and Normalize."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Simple transforms for validation/test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# Model Loading
# ============================================================

def load_model(model_name: str, num_classes: int, config: dict):
    """
    Load and configure model.

    Returns:
        (model, model_type, processor_or_none)
        model_type is 'pytorch' or 'huggingface'
    """
    logger.info(f"Loading model: {model_name}")

    if model_name == 'vit_b16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model, 'pytorch', None

    elif model_name == 'convnext_tiny':
        from transformers import ConvNextForImageClassification, ConvNextImageProcessor
        processor = ConvNextImageProcessor.from_pretrained('facebook/convnext-tiny-224')
        model = ConvNextForImageClassification.from_pretrained(
            'facebook/convnext-tiny-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        return model, 'huggingface', processor

    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 'pytorch', None

    elif model_name == 'resnest50':
        try:
            import timm
            model = timm.create_model('resnest50d', pretrained=True, num_classes=num_classes)
            return model, 'pytorch', None
        except ImportError:
            logger.error("timm package required for ResNeSt. Install: pip install timm")
            raise

    elif model_name == 'cvt':
        from transformers import CvtForImageClassification, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained('microsoft/cvt-13')
        model = CvtForImageClassification.from_pretrained(
            'microsoft/cvt-13',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        return model, 'huggingface', processor

    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Choose from: vit_b16, convnext_tiny, resnext50, resnest50, cvt")


# ============================================================
# Training with Mixed Precision + Gradient Accumulation
# ============================================================

class EnhancedTrainer(TrainerBase):
    """
    Extended TrainerBase with:
    - Mixed precision (fp16) via torch.cuda.amp
    - Gradient accumulation
    """

    def __init__(self, model, device, criterion, optimizer, scheduler=None,
                 use_fp16=False, grad_accum_steps=1):
        super().__init__(model, device, criterion, optimizer, scheduler)
        self.use_fp16 = use_fp16 and device.type == 'cuda'
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.scaler = GradScaler(enabled=self.use_fp16)

        if self.use_fp16:
            logger.info("Mixed precision (fp16) enabled")
        if self.grad_accum_steps > 1:
            logger.info(f"Gradient accumulation: {self.grad_accum_steps} steps")

    def train_epoch(self, train_loader: DataLoader):
        """Train one epoch with fp16 + gradient accumulation."""
        from tqdm import tqdm

        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        self.optimizer.zero_grad()

        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.use_fp16):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps  # scale loss

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.grad_accum_steps * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader):
        """Validate with fp16 for speed."""
        from tqdm import tqdm

        self.model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_fp16):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)
        return val_loss, val_acc


# ============================================================
# Main Training Pipeline
# ============================================================

def train_single_model(model_name: str):
    """Train a single model end-to-end."""
    fix_windows_encoding()

    config = get_config(model_name)
    device = get_device()
    img_size = config.get('img_size', 224)

    print("=" * 60)
    print(f"POULTRY DISEASE CLASSIFICATION - {model_name.upper()}")
    print("=" * 60)
    print(f"  Device        : {config['device']}")
    print(f"  Model         : {model_name}")
    print(f"  Batch size    : {config['batch_size']}")
    print(f"  Epochs        : {config['epochs']}")
    print(f"  Learning rate : {config['learning_rate']}")
    print(f"  FP16          : {config.get('fp16', False)}")
    print(f"  Grad accum    : {config.get('gradient_accumulation_steps', 1)}")
    print(f"  Focal Loss    : {config.get('use_focal_loss', False)}")
    print(f"  Weighted Samp.: {config.get('use_weighted_sampler', False)}")

    # --- Data ---
    data_dir = config['data_dir']
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("Run rebuild_dataset.py first to create the clean split!")
        return None, None

    print(f"\n  Data dir: {data_dir}")
    data_info = prepare_datasets(data_dir)
    print_dataset_info(data_info)

    classes = data_info['classes']
    num_classes = len(classes)
    label2id = data_info['label2id']
    id2label = data_info['id2label']

    # --- Class weights for loss ---
    train_labels = data_info['train']['labels']
    class_weights = get_class_weights_tensor(train_labels, num_classes).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # --- Loss function ---
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config.get('focal_gamma', 2.0)
        )
        logger.info(f"Using FocalLoss (gamma={config.get('focal_gamma', 2.0)})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using weighted CrossEntropyLoss")

    # --- Model ---
    model, model_type, processor = load_model(model_name, num_classes, config)

    # --- Datasets & DataLoaders ---
    if model_type == 'pytorch':
        train_transform = get_train_transforms_with_tensor(img_size)
        val_transform = get_val_transforms(img_size)

        train_dataset = PoultryImageDataset(
            data_info['train']['paths'], data_info['train']['labels'],
            transform=train_transform
        )
        val_dataset = PoultryImageDataset(
            data_info['val']['paths'], data_info['val']['labels'],
            transform=val_transform
        )
        test_dataset = PoultryImageDataset(
            data_info['test']['paths'], data_info['test']['labels'],
            transform=val_transform
        )

        datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        pin_memory = config.get('pin_memory', False) and torch.cuda.is_available()
        use_ws = config.get('use_weighted_sampler', False)

        train_loader, val_loader, test_loader = create_dataloaders(
            datasets,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            use_weighted_sampler=use_ws,
            pin_memory=pin_memory
        )

        # --- Train (PyTorch loop) ---
        model = model.to(device)

        optimizer, scheduler = setup_optimizer_scheduler(
            model, config['learning_rate'], config['weight_decay']
        )

        trainer = EnhancedTrainer(
            model, device, criterion, optimizer, scheduler,
            use_fp16=config.get('fp16', False),
            grad_accum_steps=config.get('gradient_accumulation_steps', 1)
        )

        history = trainer.train(
            train_loader, val_loader,
            config['epochs'], config['output_dir']
        )

        preds, labels = trainer.evaluate(test_loader, classes, config['output_dir'])
        plot_training_history(history, config['output_dir'])
        test_acc = trainer.best_acc.item() if hasattr(trainer.best_acc, 'item') else float(trainer.best_acc)

    elif model_type == 'huggingface':
        # HuggingFace models (ConvNeXt, CvT)
        train_aug = get_train_transforms(img_size)  # PIL-level augmentation (no ToTensor)

        train_dataset = HuggingFaceDataset(
            data_info['train']['paths'],
            [id2label[l] for l in data_info['train']['labels']],
            processor=processor, feature_extractor=None,
            label2id=label2id, transform=train_aug
        )
        val_dataset = HuggingFaceDataset(
            data_info['val']['paths'],
            [id2label[l] for l in data_info['val']['labels']],
            processor=processor, feature_extractor=None,
            label2id=label2id, transform=None
        )
        test_dataset = HuggingFaceDataset(
            data_info['test']['paths'],
            [id2label[l] for l in data_info['test']['labels']],
            processor=processor, feature_extractor=None,
            label2id=label2id, transform=None
        )

        # Set id2label/label2id on model config
        model.config.id2label = id2label
        model.config.label2id = label2id

        # Create FocalLoss for HF Trainer
        hf_loss = criterion if config.get('use_focal_loss', False) else None

        hf_trainer = HuggingFaceTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=config['output_dir'],
            config=config,
            loss_fn=hf_loss
        )

        hf_trainer.train()
        preds, labels = hf_trainer.evaluate(test_dataset, classes)

        from sklearn.metrics import accuracy_score
        test_acc = accuracy_score(labels, preds)
        history = {}

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # --- Summary ---
    print_summary(model_name, test_acc, config['output_dir'], classes)

    results = {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'num_classes': num_classes,
        'classes': classes,
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str, list)) else v
                   for k, v in config.items()},
    }
    save_results(results, config['output_dir'])

    return model, results


def main():
    parser = argparse.ArgumentParser(
        description='Train poultry disease classification model',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--model', type=str, default='vit_b16',
        choices=['vit_b16', 'convnext_tiny', 'resnext50', 'resnest50', 'cvt', 'all'],
        help=(
            'Model architecture to train:\n'
            '  vit_b16       - Vision Transformer B/16\n'
            '  convnext_tiny - ConvNeXt Tiny\n'
            '  resnext50     - ResNeXt-50 32x4d\n'
            '  resnest50     - ResNeSt-50d\n'
            '  cvt           - CvT-13\n'
            '  all           - Train all models sequentially'
        )
    )
    args = parser.parse_args()

    if args.model == 'all':
        all_models = ['vit_b16', 'convnext_tiny', 'resnext50', 'resnest50', 'cvt']
        results_summary = {}
        for m in all_models:
            print(f"\n{'#' * 60}")
            print(f"# Training: {m}")
            print(f"{'#' * 60}\n")
            try:
                _, result = train_single_model(m)
                if result:
                    results_summary[m] = result.get('test_accuracy', 'N/A')
            except Exception as e:
                logger.error(f"Failed to train {m}: {e}")
                results_summary[m] = f"FAILED: {e}"

        print("\n" + "=" * 60)
        print("ALL MODELS TRAINING SUMMARY")
        print("=" * 60)
        for m, acc in results_summary.items():
            if isinstance(acc, float):
                print(f"  {m:<20} Test Acc: {acc*100:.2f}%")
            else:
                print(f"  {m:<20} {acc}")
        print("=" * 60)
    else:
        train_single_model(args.model)


if __name__ == '__main__':
    main()
