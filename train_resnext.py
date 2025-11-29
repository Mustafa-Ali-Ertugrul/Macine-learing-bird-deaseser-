#!/usr/bin/env python3
"""
ResNeXt-50 (32x4d) Training Script for Poultry Disease Classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
import copy
import time
from tqdm import tqdm

# === Configuration ===
CONFIG = {
    'data_dir': 'Macine learing (bird deaseser)/final_poultry_dataset_10_classes',
    'output_dir': './resnext_poultry_results',
    'img_size': 224,
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_workers': 0,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Set seeds
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

def main():
    print("=" * 60)
    print("POULTRY DISEASE CLASSIFICATION - ResNeXt-50")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")

    # 1. Data Preparation
    data_dir = CONFIG['data_dir']
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(classes)} classes: {classes}")

    image_paths = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        cls_images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.extend(cls_images)
        labels.extend([cls] * len(cls_images))

    # Mappings
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}

    # Splits
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=CONFIG['test_size'], stratify=labels, random_state=CONFIG['random_seed']
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=CONFIG['val_size'], stratify=train_val_labels, random_state=CONFIG['random_seed']
    )

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # Dataset Class
    class PoultryDataset(Dataset):
        def __init__(self, paths, labels, transform=None):
            self.paths = paths
            self.labels = labels
            self.transform = transform
            self.label2id = label2id

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            label = self.labels[idx]
            try:
                image = Image.open(path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, self.label2id[label]
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return torch.zeros((3, CONFIG['img_size'], CONFIG['img_size'])), self.label2id[label]

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DataLoaders
    train_loader = DataLoader(PoultryDataset(train_paths, train_labels, train_transform), 
                              batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(PoultryDataset(val_paths, val_labels, val_transform), 
                            batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(PoultryDataset(test_paths, test_labels, val_transform), 
                             batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # 2. Model Setup
    print("\nLoading ResNeXt-50...")
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    
    # Modify final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
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

        epoch_loss = running_loss / len(train_paths)
        epoch_acc = running_corrects.double() / len(train_paths)

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

        val_loss = val_loss / len(val_paths)
        val_acc = val_corrects.double() / len(val_paths)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_resnext.pth'))
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
    plt.title('Confusion Matrix - ResNeXt-50')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'))
    print("✅ Saved confusion matrix")

if __name__ == '__main__':
    main()
