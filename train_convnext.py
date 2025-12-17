#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt-Tiny Training Script for Poultry Disease Classification
Using Hugging Face Transformers
"""

import os
import sys
import torch

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import ConvNextForImageClassification, ConvNextImageProcessor, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ===
CONFIG = {
    'data_dir': 'Macine learing (bird deaseser)/final_dataset_10_classes',
    'output_dir': './convnext_poultry_results',
    'model_name': 'facebook/convnext-tiny-224',
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,
    'num_workers': 0
}

# Set seeds
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

def main():
    print("=" * 60)
    print("POULTRY DISEASE CLASSIFICATION - ConvNeXt-Tiny")
    print("=" * 60)

    # 1. Data Preparation
    data_dir = os.path.join(CONFIG['data_dir'].replace('final_dataset_10_classes', 'final_dataset_split'))
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run organize_dataset_splits_physically.py first.")
        return
    
    # Load Processor FIRST
    processor = ConvNextImageProcessor.from_pretrained(CONFIG['model_name'])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')


    # Helper to collect paths ensuring consistent classes
    def collect_paths(directory, classes):
        paths = []
        labels = []
        # Use provided classes list to ensure consistency
        for i, cls in enumerate(classes):
            cls_path = os.path.join(directory, cls)
            if not os.path.exists(cls_path):
                continue
            
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'))]
            paths.extend(files)
            labels.extend([i] * len(files))
        return paths, labels

    # Detect ALL classes from TRAIN directory (Source of Truth)
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    label2id = {label: i for i, label in enumerate(train_classes)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Collect paths
    train_paths, train_labels = collect_paths(train_dir, train_classes)
    val_paths, val_labels = collect_paths(val_dir, train_classes)
    test_paths, test_labels = collect_paths(test_dir, train_classes)
    
    # Custom Dataset
    class PoultryHFDataset(Dataset):
        def __init__(self, image_paths, labels, processor, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.processor = processor
            self.transform = transform
            self.classes = train_classes # Use global classes
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new('RGB', (224, 224), color='black')
                
            if hasattr(self, 'transform') and self.transform:
                image = self.transform(image)
                
            encoding = self.processor(image, return_tensors="pt")
            item = {k: v.squeeze() for k, v in encoding.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
    
    print(f"Found {len(train_classes)} classes: {train_classes}")
    
    # Augmentation for Training
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    ])
    
    train_dataset = PoultryHFDataset(train_paths, train_labels, processor, transform=train_transforms)
    val_dataset = PoultryHFDataset(val_paths, val_labels, processor)
    test_dataset = PoultryHFDataset(test_paths, test_labels, processor)
    
    print(f"Found {len(train_classes)} classes: {train_classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 2. Model Setup
    print(f"\nLoading {CONFIG['model_name']}...")
    model = ConvNextForImageClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(train_classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # 3. Training
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        dataloader_num_workers=CONFIG['num_workers'],
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': (predictions == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\nStarting Training...")
    trainer.train()

    # Save
    trainer.save_model(os.path.join(CONFIG['output_dir'], 'final_model'))
    processor.save_pretrained(os.path.join(CONFIG['output_dir'], 'final_model'))
    print("✅ Model saved")

    # 4. Evaluation
    print("\nEvaluating on Test Set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=train_classes))

    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ConvNeXt-Tiny')
    plt.colorbar()
    tick_marks = np.arange(len(train_classes))
    plt.xticks(tick_marks, train_classes, rotation=45)
    plt.yticks(tick_marks, train_classes)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'))
    print("✅ Saved confusion matrix")

if __name__ == '__main__':
    main()
