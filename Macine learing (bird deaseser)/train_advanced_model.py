"""
Advanced Poultry Disease Classification Model Training Pipeline
================================================================
This script trains a deep learning model for classifying poultry diseases
using the final_dataset_10_classes directory structure.

Features:
- Multiple model architectures (ResNet, EfficientNet, ConvNeXt)
- Class-weighted loss for imbalanced datasets
- Learning rate scheduling with warmup
- Early stopping with patience
- Comprehensive evaluation metrics
- Model checkpointing
- TensorBoard logging support
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    'dataset_dir': 'final_dataset_10_classes',
    'output_dir': 'training_output',
    'model_name': 'resnet50',  # Options: resnet18, resnet50, efficientnet_b0, efficientnet_b2, convnext_tiny
    'num_epochs': 30,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'image_size': 224,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'patience': 7,
    'use_class_weights': True,
    'use_weighted_sampler': False,
    'num_workers': 0 if os.name == 'nt' else 4,
    'seed': 42,
    'mixed_precision': True,
}

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PoultryDiseaseDataset(Dataset):
    """Custom Dataset for Poultry Disease Classification"""
    
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size']), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_training=True):
    """Get data transforms for training and validation"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'] + 32, CONFIG['image_size'] + 32)),
            transforms.RandomCrop(CONFIG['image_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

def load_dataset():
    """Load dataset from directory structure"""
    dataset_dir = Path(CONFIG['dataset_dir'])
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    image_paths = []
    labels = []
    class_names = []
    
    # Get all class directories
    for class_dir in sorted(dataset_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Get all images in this class
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    image_paths.append(str(img_path))
                    labels.append(class_name)
    
    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_names))}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Convert labels to indices
    label_indices = [class_to_idx[label] for label in labels]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {len(image_paths)}")
    print(f"   Number of classes: {len(class_names)}")
    print(f"\n   Class distribution:")
    
    class_counts = Counter(labels)
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = count / len(labels) * 100
        print(f"   {cls:30} : {count:5} ({pct:5.1f}%)")
    
    return image_paths, label_indices, class_to_idx, idx_to_class

def create_data_splits(image_paths, labels, class_to_idx):
    """Create train/val/test splits with stratification"""
    
    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(1 - CONFIG['train_ratio']),
        random_state=CONFIG['seed'],
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio_adjusted = CONFIG['val_ratio'] / (CONFIG['val_ratio'] + CONFIG['test_ratio'])
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        random_state=CONFIG['seed'],
        stratify=temp_labels
    )
    
    print(f"\n📊 Data Splits:")
    print(f"   Train: {len(train_paths)} images")
    print(f"   Validation: {len(val_paths)} images")
    print(f"   Test: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = PoultryDiseaseDataset(
        train_paths, train_labels, class_to_idx,
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = PoultryDiseaseDataset(
        val_paths, val_labels, class_to_idx,
        transform=get_transforms(is_training=False)
    )
    
    test_dataset = PoultryDiseaseDataset(
        test_paths, test_labels, class_to_idx,
        transform=get_transforms(is_training=False)
    )
    
    return train_dataset, val_dataset, test_dataset, train_labels

def get_model(num_classes):
    """Get the model architecture"""
    model_name = CONFIG['model_name']
    
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return epoch_loss, epoch_acc, f1, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved to: {output_path}")

def plot_training_history(history, output_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1 Score
    axes[2].plot(history['val_f1'], label='Validation F1')
    axes[2].set_title('F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score (%)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Training history saved to: {output_path}")

def main():
    print("=" * 70)
    print("POULTRY DISEASE CLASSIFICATION - ADVANCED TRAINING PIPELINE")
    print("=" * 70)
    
    # Set seed
    set_seed(CONFIG['seed'])
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    image_paths, labels, class_to_idx, idx_to_class = load_dataset()
    num_classes = len(class_to_idx)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    # Create data splits
    print("\n📊 Creating data splits...")
    train_dataset, val_dataset, test_dataset, train_labels = create_data_splits(
        image_paths, labels, class_to_idx
    )
    
    # Calculate class weights
    class_weights = None
    if CONFIG['use_class_weights']:
        class_weights = calculate_class_weights(train_labels).to(device)
        print(f"\n⚖️ Class weights calculated for imbalanced dataset")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print(f"\n🧠 Creating model: {CONFIG['model_name']}")
    model = get_model(num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] and torch.cuda.is_available() else None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {CONFIG['num_epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    print("-" * 70)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:2d}/{CONFIG['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.2f}% | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            epochs_no_improve = 0
            
            # Save model
            model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_to_idx': class_to_idx,
                'config': CONFIG
            }, model_path)
            print(f"   ✅ Best model saved! (F1: {val_f1:.2f}%)")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= CONFIG['patience']:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print("-" * 70)
    print(f"\n🎉 Training completed in {total_time/60:.1f} minutes")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Best validation F1 score: {best_val_f1:.2f}%")
    
    # Load best model for evaluation
    print("\n📊 Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    print(f"\n📈 Test Results:")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   F1 Score: {test_f1:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Save confusion matrix
    plot_confusion_matrix(test_labels, test_preds, class_names, output_dir / 'confusion_matrix.png')
    
    # Save training history
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Save results to JSON
    results = {
        'config': CONFIG,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'training_time_minutes': total_time / 60,
        'class_names': class_names,
        'history': history
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ All results saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
