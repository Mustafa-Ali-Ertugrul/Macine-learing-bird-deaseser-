#!/usr/bin/env python3
"""
Vision Transformer (ViT-B/16) Fine-tuning for Poultry Disease Classification
Using Hugging Face Transformers with 10 disease categories
"""

import os
import torch
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
    'data_dir': 'organized_labeled_dataset',  # CSV-labeled and organized dataset
    'output_dir': './vit_poultry_results',
    'img_size': 224,  # ViT-B/16 standard input size
    'batch_size': 16,  # Adjust based on GPU memory (8 for 6GB, 16 for 8GB+)
    'epochs': 15,
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
data_dir = CONFIG['data_dir']
if not os.path.exists(data_dir):
    print(f"❌ Data directory not found: {data_dir}")
    print("Please run resize_images_512.py first!")
    exit(1)

classes = sorted([d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))])

print(f"\n📊 Found {len(classes)} disease categories:")
for i, cls in enumerate(classes, 1):
    print(f"   {i}. {cls}")

# Collect all images
image_paths = []
labels = []

print(f"\n📁 Collecting images from {data_dir}...")
for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    cls_images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    
    image_paths.extend(cls_images)
    labels.extend([cls] * len(cls_images))
    print(f"   {cls}: {len(cls_images)} images")

print(f"\n✅ Total images: {len(image_paths)}")

if len(image_paths) == 0:
    print("❌ No images found! Check your data directory.")
    exit(1)

# === 2. Train/Val/Test Split ===
print("\n📊 Splitting dataset...")

# First split: train+val vs test
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    image_paths, labels, 
    test_size=CONFIG['test_size'], 
    stratify=labels, 
    random_state=CONFIG['random_seed']
)

# Second split: train vs val
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels,
    test_size=CONFIG['val_size'],
    stratify=train_val_labels,
    random_state=CONFIG['random_seed']
)

print(f"   Training: {len(train_paths)} images")
print(f"   Validation: {len(val_paths)} images")
print(f"   Test: {len(test_paths)} images")

# === 3. Custom Dataset for ViT ===
class PoultryViTDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor, label2id):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.label2id = label2id
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # Return a blank image if error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Process with feature extractor
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        
        # Prepare item
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = self.label2id[self.labels[idx]]
        
        return item

# === 4. Load ViT Model ===
print("\n🔮 Loading ViT-B/16 from Hugging Face...")
print("   (First run will download ~1GB model)")

# Create label mappings
label2id = {label: i for i, label in enumerate(sorted(set(labels)))}
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
print("\n📦 Creating datasets...")
train_dataset = PoultryViTDataset(train_paths, train_labels, feature_extractor, label2id)
val_dataset = PoultryViTDataset(val_paths, val_labels, feature_extractor, label2id)
test_dataset = PoultryViTDataset(test_paths, test_labels, feature_extractor, label2id)

print(f"   Train dataset: {len(train_dataset)} samples")
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
