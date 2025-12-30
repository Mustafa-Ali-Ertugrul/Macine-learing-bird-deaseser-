#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common Configuration for Poultry Disease Classification Training
Centralized configuration for all model training scripts
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMMON_CONFIG = {
    'data_dir': os.path.join(BASE_DIR, 'Macine learing (bird deaseser)', 'final_dataset_split'),
    'img_size': 224,
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'num_workers': 0,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 10
}

TRANSFORM_CONFIG = {
    'train': {
        'resize': 224,
        'random_horizontal_flip': 0.5,
        'random_rotation': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        },
        'random_resized_crop': {
            'scale': (0.8, 1.0),
            'ratio': (0.9, 1.1)
        }
    },
    'val': {
        'resize': 224
    }
}

IMAGENET_NORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

DISEASE_CLASSES = [
    'Avian_Influenza',
    'Coccidiosis',
    'Fowl_Pox',
    'Healthy',
    'Histomoniasis',
    'Infectious_Bronchitis',
    'Infectious_Bursal_Disease',
    'Mareks_Disease',
    'Necrotic_Enteritis',
    'Salmonellosis'
]

MODEL_CONFIGS = {
    'vit_b16': {
        'model_name': 'google/vit-base-patch16-224-in21k',
        'output_dir': './vit_poultry_results',
        'batch_size': 16
    },
    'convnext_tiny': {
        'model_name': 'facebook/convnext-tiny-224',
        'output_dir': './convnext_poultry_results',
        'batch_size': 16
    },
    'resnext50': {
        'model_name': 'resnext50_32x4d',
        'output_dir': './resnext_poultry_results',
        'batch_size': 16,
        'weights': 'IMAGENET1K_V1'
    },
    'resnest50': {
        'model_name': 'resnest50',
        'output_dir': './resnest_poultry_results',
        'batch_size': 16
    },
    'cvt': {
        'model_name': 'microsoft/cvt-21-384-224',
        'output_dir': './cvt_poultry_results',
        'batch_size': 16
    }
}

def get_config(model_name='vit_b16'):
    base_config = COMMON_CONFIG.copy()
    model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['vit_b16'])
    base_config.update(model_config)
    return base_config

def get_device():
    return torch.device(COMMON_CONFIG['device'])

def fix_windows_encoding():
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
