#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT-B/16) Fine-tuning for Poultry Disease Classification
Using Hugging Face Transformers with 10 disease categories
"""

import os
import sys
import torch

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json

# Hugging Face ViT
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

# === Configuration ===
CONFIG = {
    'data_dir': 'Macine learing (bird deaseser)/final_dataset_10_classes',  # Updated dataset path
    'output_dir': './vit_poultry_results',
    'img_size': 224,  # ViT-B/16 standard input size
    'batch_size': 16,  # Adjust based on GPU memory (8 for 6GB, 16 for 8GB+)
    'epochs': 10,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'eval_steps': 100,
    'save_steps': 100,
    'logging_steps': 50,
    'test_size': 0.2,
    'val_size': 0.1,  # From training set
    'random_seed': 42,
    'num_workers': 0,  # Set to 0 for Windows compatibility
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# === 1. Collect Dataset ===
print("=" * 60)
print("POULTRY DISEASE CLASSIFICATION - ViT-B/16")
print("=" * 60)

# Get class folders
data_dir = os.path.join(CONFIG['data_dir'].replace('final_dataset_10_classes', 'final_dataset_split'))
if not os.path.exists(data_dir):
    print(f"❌ Data directory not found: {data_dir}")
    print("Please run organize_dataset_splits_physically.py first!")
    exit(1)

print(f"Loading data from: {data_dir}")

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Helper to collect paths
def collect_paths(directory):
    paths = []
    labels = []
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        paths.extend(files)
        labels.extend([cls] * len(files))
    return paths, labels, classes

print("\n📊 Loading splits...")
train_paths, train_labels, classes = collect_paths(train_dir)
val_paths, val_labels, _ = collect_paths(val_dir)
test_paths, test_labels, _ = collect_paths(test_dir)

print(f"   Training: {len(train_paths)} images")
print(f"   Validation: {len(val_paths)} images")
print(f"   Test: {len(test_paths)} images")

print(f"\n📊 Found {len(classes)} disease categories:")
for i, cls in enumerate(classes, 1):
    print(f"   {i}. {cls}")

# === 3. Custom Dataset for ViT ===
class PoultryViTDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor, label2id, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms (Augmentation) if provided
        if hasattr(self, 'transform') and self.transform:
            image = self.transform(image)

        # Process with feature extractor
        try:
            encoding = self.feature_extractor(images=image, return_tensors="pt")
        except Exception as e:
            print(f"❌ Error processing image: {img_path}")
            print(f"   Mode: {image.mode}, Size: {image.size}")
            print(f"   Error: {e}")
            # Return a blank image to avoid crashing
            image = Image.new('RGB', (224, 224), color='black')
            encoding = self.feature_extractor(images=image, return_tensors="pt")
        
        # Prepare item
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = self.label2id[self.labels[idx]]
        
        return item

# === 4. Load ViT Model ===
print("\n🔮 Loading ViT-B/16 from Hugging Face...")
print("   (First run will download ~1GB model)")

# Create label mappings
label2id = {label: i for i, label in enumerate(classes)}
id2label = {i: label for label, i in label2id.items()}

# Save label mappings
os.makedirs(CONFIG['output_dir'], exist_ok=True)
with open(os.path.join(CONFIG['output_dir'], 'label_mappings.json'), 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# Load pre-trained model and adapt for our classes
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(classes),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

print(f"✅ Model loaded with {len(classes)} output classes")

# === 5. Create Datasets ===
# Augmentation for Training
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
])

print("\n📦 Creating datasets...")
train_dataset = PoultryViTDataset(train_paths, train_labels, feature_extractor, label2id, transform=train_transforms)
val_dataset = PoultryViTDataset(val_paths, val_labels, feature_extractor, label2id)
test_dataset = PoultryViTDataset(test_paths, test_labels, feature_extractor, label2id)

print(f"   Train dataset: {len(train_dataset)} samples (with on-the-fly augmentation)")
print(f"   Val dataset: {len(val_dataset)} samples")
print(f"   Test dataset: {len(test_dataset)} samples")

# === 6. Compute Class Weights (for imbalanced dataset) ===
from collections import Counter

class_counts = Counter(train_labels)
total_samples = len(train_labels)
class_weights = {label2id[cls]: total_samples / (len(class_counts) * count) 
                for cls, count in class_counts.items()}

print(f"\n⚖️ Class weights (for imbalanced classes):")
for cls, weight in sorted(class_weights.items()):
    print(f"   {id2label[cls]}: {weight:.2f}")

# === 7. Training Arguments ===
training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    learning_rate=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    warmup_ratio=CONFIG['warmup_ratio'],
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=CONFIG['eval_steps'],
    save_strategy="steps",
    save_steps=CONFIG['save_steps'],
    logging_steps=CONFIG['logging_steps'],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    dataloader_num_workers=CONFIG['num_workers'],
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",  # Disable wandb/tensorboard
    seed=CONFIG['random_seed'],
)

# === 8. Metrics ===
def compute_metrics(eval_pred):
    """Compute accuracy and other metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = (predictions == labels).mean()
    
    return {
        'accuracy': accuracy,
    }

# === 9. Create Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# === 10. Train ===
print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)
print(f"Device: {training_args.device}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"Epochs: {CONFIG['epochs']}")
print("=" * 60 + "\n")

trainer.train()

# === 11. Save Final Model ===
final_model_path = os.path.join(CONFIG['output_dir'], 'final_model')
trainer.save_model(final_model_path)
feature_extractor.save_pretrained(final_model_path)

print(f"\n✅ Model saved to: {final_model_path}")

# === 12. Evaluate on Test Set ===
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)

test_results = trainer.predict(test_dataset)
test_predictions = np.argmax(test_results.predictions, axis=-1)
test_true_labels = [label2id[label] for label in test_labels]

# Accuracy
test_accuracy = (test_predictions == test_true_labels).sum() / len(test_true_labels)
print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(
    test_true_labels, 
    test_predictions,
    target_names=[id2label[i] for i in range(len(classes))],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(test_true_labels, test_predictions)
print("\n📈 Confusion Matrix saved to confusion_matrix.png")

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - ViT-B/16 Poultry Disease Classification')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, [id2label[i] for i in range(len(classes))], rotation=45, ha='right')
plt.yticks(tick_marks, [id2label[i] for i in range(len(classes))])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved!")

# === 13. Summary ===
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"📁 Model saved: {final_model_path}")
print(f"📊 Results saved: {CONFIG['output_dir']}")
print(f"🎯 Test Accuracy: {test_accuracy*100:.2f}%")
print("\nNext steps:")
print("  1. Check confusion_matrix.png for per-class performance")
print("  2. Use predict_single.py for inference on new images")
print("  3. Deploy model for production use")
print("=" * 60)
