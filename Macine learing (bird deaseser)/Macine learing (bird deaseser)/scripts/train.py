"""
Advanced Poultry Disease Classification Training Script
========================================================
Refactored training pipeline using modular components.
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_factory import get_model
from src.data.dataset import PoultryDiseaseDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.augmentation import AdvancedAugmentation, compute_mixed_loss
from src.training.tta import predict_with_tta
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("training")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset_dir: str) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """
    Load dataset from directory structure.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        Tuple of (image_paths, labels, class_to_idx, idx_to_class)
    """
    dataset_dir = Path(dataset_dir)
    
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
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total images: {len(image_paths)}")
    logger.info(f"  Number of classes: {len(class_names)}")
    logger.info(f"  Class distribution:")
    
    class_counts = Counter(labels)
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = count / len(labels) * 100
        logger.info(f"    {cls:30} : {count:5} ({pct:5.1f}%)")
    
    return image_paths, label_indices, class_to_idx, idx_to_class


def create_data_splits(
    image_paths: List[str],
    labels: List[int],
    class_to_idx: Dict[str, int],
    config: Dict
) -> Tuple[PoultryDiseaseDataset, PoultryDiseaseDataset, PoultryDiseaseDataset, List[int]]:
    """Create train/val/test splits."""
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    seed = config['seed']
    image_size = config['data']['image_size']
    
    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        random_state=seed,
        stratify=temp_labels
    )
    
    logger.info(f"Data Splits:")
    logger.info(f"  Train: {len(train_paths)} images")
    logger.info(f"  Validation: {len(val_paths)} images")
    logger.info(f"  Test: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = PoultryDiseaseDataset(
        train_paths, train_labels, class_to_idx,
        transform=get_train_transforms(image_size)
    )
    
    val_dataset = PoultryDiseaseDataset(
        val_paths, val_labels, class_to_idx,
        transform=get_val_transforms(image_size)
    )
    
    test_dataset = PoultryDiseaseDataset(
        test_paths, test_labels, class_to_idx,
        transform=get_val_transforms(image_size)
    )
    
    return train_dataset, val_dataset, test_dataset, train_labels


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset."""
    class_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler=None,
    adv_aug=None,
    gradient_accumulation_steps: int = 1
) -> Tuple[float, float]:
    """Train for one epoch with optional advanced augmentation and gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        use_mix = adv_aug is not None
        labels_a, labels_b, lam = labels, labels, 1.0
        
        if use_mix:
            images, labels_a, labels_b, lam = adv_aug(images, labels)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if use_mix:
                    loss = compute_mixed_loss(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            if use_mix:
                loss = compute_mixed_loss(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if use_mix:
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                correct += (predicted == labels).sum().item()
        
        running_loss += loss.item() * gradient_accumulation_steps
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, List[int], List[int]]:
    """Validate the model."""
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


def validate_with_tta(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_augmentations: int = 5
) -> Tuple[float, float, float, List[int], List[int]]:
    """Validate the model with Test-Time Augmentation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    from src.training.tta import TTAAugmentation
    tta_aug = TTAAugmentation(num_augmentations)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            augmented_images = tta_aug(images)
            all_probs = []
            
            for aug_img in augmented_images:
                outputs = model(aug_img)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)
            
            avg_probs = torch.stack(all_probs).mean(dim=0)
            loss = criterion(avg_probs.log(), labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(avg_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return epoch_loss, epoch_acc, f1, all_preds, all_labels



def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: Path
) -> None:
    """Plot and save confusion matrix."""
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
    logger.info(f"  Confusion matrix saved to: {output_path}")


def plot_training_history(history: Dict, output_path: Path) -> None:
    """Plot training history."""
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
    logger.info(f"  Training history saved to: {output_path}")


def main():
    """Main training function."""
    logger.info("="*70)
    logger.info("POULTRY DISEASE CLASSIFICATION - TRAINING PIPELINE")
    logger.info("="*70)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    config = ConfigLoader.load(config_path)
    logger.info(f"Configuration loaded from: {config_path}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create output directory
    output_dir = Path(config['logging']['save_model_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_dir = Path(config['data']['dataset_dir'])
    if not dataset_dir.exists():
        dataset_dir = Path("final_dataset_10_classes")  # Fallback
    
    image_paths, labels, class_to_idx, idx_to_class = load_dataset(dataset_dir)
    num_classes = len(class_to_idx)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    # Create data splits
    logger.info("Creating data splits...")
    train_dataset, val_dataset, test_dataset, train_labels = create_data_splits(
        image_paths, labels, class_to_idx, config
    )
    
    # Calculate class weights
    class_weights = None
    if config['training']['loss']['use_class_weights']:
        class_weights = calculate_class_weights(train_labels, num_classes).to(device)
        logger.info("Class weights calculated for imbalanced dataset")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )
    
    # Create model
    model_name = config['model']['architecture']
    use_enhanced_head = config['model'].get('use_enhanced_head', True)
    logger.info(f"Creating model: {model_name}")
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        use_enhanced_head=use_enhanced_head
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = None
    if config['training']['mixed_precision'] and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    
    # Advanced augmentation (CutMix/MixUp/RandAugment)
    adv_aug = None
    aug_config = config.get('augmentation', {})
    if aug_config.get('use_randaugment') or aug_config.get('cutmix_prob', 0) > 0:
        adv_aug = AdvancedAugmentation(
            use_randaugment=aug_config.get('use_randaugment', False),
            randaugment_m=aug_config.get('randaugment_m', 9),
            randaugment_n=aug_config.get('randaugment_n', 2),
            cutmix_prob=aug_config.get('cutmix_prob', 0.5),
            mixup_prob=aug_config.get('mixup_prob', 0.3),
            cutmix_alpha=aug_config.get('cutmix_alpha', 1.0),
            mixup_alpha=aug_config.get('mixup_alpha', 0.4)
        )
        logger.info(f"Advanced augmentation enabled: RandAugment={aug_config.get('use_randaugment')}, "
                   f"CutMix={aug_config.get('cutmix_prob')}, MixUp={aug_config.get('mixup_prob')}")
    
    # Training strategy parameters
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    use_tta = config['training'].get('use_tta', False)
    tta_num = config['training'].get('tta_num_aug', 5)
    base_lr = config['training']['optimizer']['lr']
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"Training strategy: Gradient Accumulation = {gradient_accumulation_steps}, "
               f"Effective batch size = {effective_batch_size}, Warmup epochs = {warmup_epochs}, "
               f"TTA = {use_tta}")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    patience = config['training']['early_stopping']['patience']
    
    logger.info("Starting training...")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {config['training']['optimizer']['lr']}")
    logger.info("-"*70)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Learning rate warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            logger.info(f"Warmup: LR = {warmup_lr:.2e}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, adv_aug,
            gradient_accumulation_steps
        )
        
        # Validate (with TTA if enabled)
        if use_tta and epoch >= num_epochs - 3:
            logger.info(f"Using TTA for validation at epoch {epoch+1}")
            val_loss, val_acc, val_f1, val_preds, val_labels = validate_with_tta(
                model, val_loader, criterion, device, tta_num
            )
        else:
            val_loss, val_acc, val_f1, val_preds, val_labels = validate(
                model, val_loader, criterion, device
            )
        
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
        
        logger.info(
            f"Epoch [{epoch+1:2d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )
        
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
                'config': config
            }, model_path)
            logger.info(f"  ✅ Best model saved! (F1: {val_f1:.2f}%)")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if epochs_no_improve >= patience:
                logger.info(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                break
    
    total_time = time.time() - start_time
    logger.info("-"*70)
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    logger.info(f"  Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"  Best validation F1 score: {best_val_f1:.2f}%")
    
    # Load best model for evaluation
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation with TTA if enabled
    if use_tta:
        logger.info(f"Using TTA ({tta_num} augmentations) for test evaluation...")
        test_loss, test_acc, test_f1, test_preds, test_labels = validate_with_tta(
            model, test_loader, criterion, device, tta_num
        )
    else:
        test_loss, test_acc, test_f1, test_preds, test_labels = validate(
            model, test_loader, criterion, device
        )
    
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {test_acc:.2f}%")
    logger.info(f"  F1 Score: {test_f1:.2f}%")
    
    # Classification report
    logger.info("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Save confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        output_dir / 'confusion_matrix.png'
    )
    
    # Save training history
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Save results to JSON
    results = {
        'config': config,
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
    
    logger.info(f"All results saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
