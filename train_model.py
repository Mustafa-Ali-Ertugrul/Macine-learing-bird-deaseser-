#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal Training Script for Poultry Disease Classification
Supports multiple models: ViT, ConvNeXt, ResNeXt, ResNeSt, CVT
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import ConvNextForImageClassification, ConvNextImageProcessor
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import get_config, fix_windows_encoding, get_device, DISEASE_CLASSES
from src.dataset_utils import (
    PoultryImageDataset, HuggingFaceDataset, prepare_datasets,
    print_dataset_info, get_transforms, create_label_mappings
)
from src.training_utils import TrainerBase, HuggingFaceTrainer, print_summary

def load_model(model_name='vit_b16', num_classes=10):
    """Load model based on type"""
    if model_name == 'vit_b16':
        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model, 'pytorch'
    
    elif model_name == 'convnext_tiny':
        model = ConvNextForImageClassification.from_pretrained(
            'facebook/convnext-tiny-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        return model, 'huggingface'
    
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 'pytorch'
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def create_datasets(data_info, config, model_type='pytorch'):
    """Create datasets based on model type"""
    if model_type == 'huggingface':
        return None
    else:
        train_transform = get_transforms('train', config)
        val_transform = get_transforms('val', config)
        
        train_dataset = PoultryImageDataset(
            data_info['train']['paths'],
            data_info['train']['labels'],
            transform=train_transform
        )
        val_dataset = PoultryImageDataset(
            data_info['val']['paths'],
            data_info['val']['labels'],
            transform=val_transform
        )
        test_dataset = PoultryImageDataset(
            data_info['test']['paths'],
            data_info['test']['labels'],
            transform=val_transform
        )
        
        return train_dataset, val_dataset, test_dataset

def train_model(model_name='vit_b16'):
    """Train model"""
    fix_windows_encoding()
    
    config = get_config(model_name)
    device = get_device()
    
    print("=" * 60)
    print(f"POULTRY DISEASE CLASSIFICATION - {model_name.upper()}")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Model: {model_name}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    if not os.path.exists(config['data_dir']):
        print(f"❌ Data directory not found: {config['data_dir']}")
        print("Please run organize_dataset_splits_physically.py first!")
        return
    
    print(f"\n📁 Loading data from: {config['data_dir']}")
    data_info = prepare_datasets(config['data_dir'])
    print_dataset_info(data_info)
    
    model, model_type = load_model(model_name, len(data_info['classes']))
    model = model.to(device)
    
    print(f"\n🔮 Model loaded: {model_name} ({model_type})")
    
    if model_type == 'pytorch':
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), 
                                     lr=config['learning_rate'], 
                                     weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        
        trainer = TrainerBase(model, device, criterion, optimizer, scheduler)
        
        train_dataset, val_dataset, test_dataset = create_datasets(data_info, config, model_type)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 shuffle=True, num_workers=config['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=config['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                shuffle=False, num_workers=config['num_workers'])
        
        history = trainer.train(train_loader, val_loader, config['epochs'], config['output_dir'])
        preds, labels = trainer.evaluate(test_loader, data_info['classes'], config['output_dir'])
        
        from src.training_utils import plot_training_history
        plot_training_history(history, config['output_dir'])
        
        test_acc = trainer.best_acc.item()
        
    else:
        raise NotImplementedError("HuggingFace training not yet implemented in this version")
    
    print_summary(model_name, test_acc, config['output_dir'], data_info['classes'])
    
    results = {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'history': history,
        'config': config
    }
    
    from src.training_utils import save_results
    save_results(results, config['output_dir'])
    
    return model, results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train poultry disease classification model')
    parser.add_argument('--model', type=str, default='vit_b16',
                       choices=['vit_b16', 'convnext_tiny', 'resnext50'],
                       help='Model architecture to train')
    args = parser.parse_args()
    
    train_model(args.model)
