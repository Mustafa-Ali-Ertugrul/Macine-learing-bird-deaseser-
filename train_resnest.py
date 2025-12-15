#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNeSt-50d Training Script for Poultry Disease Classification
Using `timm` library
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# === Configuration ===
CONFIG = {
    'data_dir': 'Macine learing (bird deaseser)/final_dataset_10_classes',
    'output_dir': './resnest_poultry_results',
    'img_size': 224,
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_workers': 0,
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Set seeds
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

def main():
    print("=" * 60)
    print("POULTRY DISEASE CLASSIFICATION - ResNeSt-50d")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")

    # 1. Data Preparation
    data_dir = os.path.join(CONFIG['data_dir'].replace('final_dataset_10_classes', 'final_dataset_split'))
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run organize_dataset_splits_physically.py first.")
        return

    print(f"Loading data from: {data_dir}")

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Helper to collect paths ensuring consistent classes
    def collect_paths(directory, classes):
        paths = []
        labels = []
        for i, cls in enumerate(classes):
            cls_path = os.path.join(directory, cls)
            if not os.path.exists(cls_path):
                continue
            
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            paths.extend(files)
            labels.extend([i] * len(files))
        return paths, labels

    # Detect classes from Train (Source of Truth)
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    # Collect paths
    train_paths, train_labels = collect_paths(train_dir, classes)
    val_paths, val_labels = collect_paths(val_dir, classes)
    test_paths, test_labels = collect_paths(test_dir, classes)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom Dataset
    class PoultryDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            self.classes = classes
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new('RGB', (224, 224), color='black') # Fallback

            if self.transform:
                image = self.transform(image)
            
            return image, self.labels[idx]

    train_dataset = PoultryDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PoultryDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = PoultryDataset(test_paths, test_labels, transform=val_transform)
    
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # 2. Model Setup
    print("\nLoading ResNeSt-50d...")
    model = timm.create_model('resnest50d', pretrained=True, num_classes=len(classes))
    model = model.to(CONFIG['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # 3. Training Loop
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 10)

        # Train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validate
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_resnest.pth'))
            print("✅ Saved best model")

    print(f"\nBest Val Acc: {best_acc:.4f}")

    # 4. Final Evaluation
    print("\nEvaluating on Test Set...")
    model.load_state_dict(best_model_wts)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(CONFIG['device'])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ResNeSt-50d')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'))
    print("✅ Saved confusion matrix")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
