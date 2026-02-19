#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common Dataset Utilities for Poultry Disease Classification
Provides reusable dataset loading and preprocessing functions
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter

class PoultryImageDataset(Dataset):
    """Standard dataset for poultry disease classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

class HuggingFaceDataset(Dataset):
    """Dataset for Hugging Face models (ViT, ConvNeXt, CVT)"""
    
    def __init__(self, image_paths: List[str], labels: List[str], 
                 processor, feature_extractor, label2id: dict, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        encoding = self.processor(image, return_tensors="pt") if self.processor else \
                   self.feature_extractor(images=image, return_tensors="pt")
        
        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = self.label2id[self.labels[idx]]
        return item

def collect_paths(directory: str, classes: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[str]]:
    """Collect image paths and labels from directory"""
    paths = []
    labels = []
    
    if classes is None:
        classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        if not os.path.exists(cls_path):
            continue
        
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'))]
        paths.extend(files)
        labels.extend([cls] * len(files))
    
    return paths, labels, classes

def get_transforms(transform_type='train', config=None):
    """Get image transforms based on type"""
    normalize_mean = config['mean'] if config else [0.485, 0.456, 0.406]
    normalize_std = config['std'] if config else [0.229, 0.224, 0.225]
    
    if transform_type == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

def create_label_mappings(classes: List[str]) -> Tuple[dict, dict]:
    """Create label2id and id2label mappings"""
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

def calculate_class_weights(labels: List[str]) -> dict:
    """Calculate class weights for imbalanced dataset"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    label2id, _ = create_label_mappings(sorted(class_counts.keys()))
    
    class_weights = {
        label2id[cls]: total_samples / (len(class_counts) * count) 
        for cls, count in class_counts.items()
    }
    
    return class_weights

def prepare_datasets(data_dir: str, config=None):
    """Prepare train, validation, and test datasets"""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        raise FileNotFoundError(f"Data split directories not found in {data_dir}")
    
    train_paths, train_labels, classes = collect_paths(train_dir)
    val_paths, val_labels, _ = collect_paths(val_dir, classes)
    test_paths, test_labels, _ = collect_paths(test_dir, classes)
    
    label2id, id2label = create_label_mappings(classes)
    
    train_labels_ids = [label2id[label] for label in train_labels]
    val_labels_ids = [label2id[label] for label in val_labels]
    test_labels_ids = [label2id[label] for label in test_labels]
    
    return {
        'train': {'paths': train_paths, 'labels': train_labels_ids},
        'val': {'paths': val_paths, 'labels': val_labels_ids},
        'test': {'paths': test_paths, 'labels': test_labels_ids},
        'classes': classes,
        'label2id': label2id,
        'id2label': id2label
    }

def create_dataloaders(datasets, batch_size=16, num_workers=0, shuffle_train=True):
    """Create dataloaders for train, val, and test"""
    train_loader = DataLoader(
        datasets['train'], 
        batch_size=batch_size, 
        shuffle=shuffle_train, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        datasets['val'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        datasets['test'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def print_dataset_info(data_info):
    """Print dataset information"""
    print(f"\n📊 Dataset Info:")
    print(f"   Training: {len(data_info['train']['paths'])} images")
    print(f"   Validation: {len(data_info['val']['paths'])} images")
    print(f"   Test: {len(data_info['test']['paths'])} images")
    print(f"\n📊 Classes: {len(data_info['classes'])}")
    for i, cls in enumerate(data_info['classes'], 1):
        train_count = data_info['train']['labels'].count(data_info['label2id'][cls])
        print(f"   {i}. {cls}: {train_count} training images")
