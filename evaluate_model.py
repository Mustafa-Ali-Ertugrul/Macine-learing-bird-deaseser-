#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained poultry disease classification model on the test set.

Supports all model types: vit_b16, convnext_tiny, resnext50, resnest50, cvt

Usage:
    python evaluate_model.py --model vit_b16
    python evaluate_model.py --model convnext_tiny
    python evaluate_model.py --model resnext50 --model-path ./resnext_poultry_results/best_model.pth
"""

import os
import sys
import argparse
import json
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config, fix_windows_encoding, get_device, COMMON_CONFIG
from src.dataset_utils import (
    PoultryImageDataset, HuggingFaceDataset, prepare_datasets,
    create_label_mappings
)
from src.training_utils import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_test_transforms(img_size=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_pytorch_model(model_name, num_classes, model_path, device):
    """Load a PyTorch model from saved weights."""
    from torchvision import models

    if model_name == 'vit_b16':
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnest50':
        import timm
        model = timm.create_model('resnest50d', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown pytorch model: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_pytorch(model, test_loader, classes, device, output_dir):
    """Evaluate a PyTorch model."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:\n{report}")

    plot_confusion_matrix(all_labels, all_preds, classes, output_dir, title="Test Confusion Matrix")

    return accuracy, report


def evaluate_huggingface(model_name, model_path, test_paths, test_labels_str,
                         label2id, classes, output_dir):
    """Evaluate a HuggingFace model."""
    if model_name == 'convnext_tiny':
        from transformers import ConvNextForImageClassification, ConvNextImageProcessor
        processor = ConvNextImageProcessor.from_pretrained(model_path)
        model = ConvNextForImageClassification.from_pretrained(model_path)
    elif model_name == 'cvt':
        from transformers import CvtForImageClassification, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = CvtForImageClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown HF model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = HuggingFaceDataset(
        test_paths, test_labels_str,
        processor=processor, feature_extractor=None,
        label2id=label2id, transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            labels_batch = batch.pop('labels')
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:\n{report}")

    plot_confusion_matrix(all_labels, all_preds, classes, output_dir, title="Test Confusion Matrix")

    return accuracy, report


def main():
    fix_windows_encoding()

    parser = argparse.ArgumentParser(description='Evaluate poultry disease model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['vit_b16', 'convnext_tiny', 'resnext50', 'resnest50', 'cvt'],
                        help='Model architecture')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights (auto-detected if not specified)')
    args = parser.parse_args()

    config = get_config(args.model)
    device = get_device()
    output_dir = config['output_dir']

    # Determine model type
    hf_models = {'convnext_tiny', 'cvt'}
    pytorch_models = {'vit_b16', 'resnext50', 'resnest50'}

    # Auto-detect model path
    if args.model_path:
        model_path = args.model_path
    elif args.model in hf_models:
        model_path = os.path.join(output_dir, 'final_model')
    else:
        model_path = os.path.join(output_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Train the model first: python train_model.py --model " + args.model)
        return

    # Load data
    data_dir = config['data_dir']
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    data_info = prepare_datasets(data_dir)
    classes = data_info['classes']
    num_classes = len(classes)
    label2id = data_info['label2id']
    id2label = data_info['id2label']

    print("=" * 60)
    print(f"EVALUATING: {args.model.upper()}")
    print(f"  Model path : {model_path}")
    print(f"  Test images: {len(data_info['test']['paths'])}")
    print(f"  Classes    : {num_classes}")
    print("=" * 60)

    if args.model in pytorch_models:
        model = load_pytorch_model(args.model, num_classes, model_path, device)
        test_transform = get_test_transforms()
        test_dataset = PoultryImageDataset(
            data_info['test']['paths'], data_info['test']['labels'],
            transform=test_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        accuracy, report = evaluate_pytorch(model, test_loader, classes, device, output_dir)
    else:
        test_labels_str = [id2label[l] for l in data_info['test']['labels']]
        accuracy, report = evaluate_huggingface(
            args.model, model_path,
            data_info['test']['paths'], test_labels_str,
            label2id, classes, output_dir
        )

    # Save report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
