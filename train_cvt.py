#!/usr/bin/env python3
"""
CvT-13 (Convolutional Vision Transformer) Training Script for Poultry Disease Classification
Using Hugging Face Transformers
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
from transformers import CvtForImageClassification, AutoImageProcessor, TrainingArguments, Trainer, EarlyStoppingCallback

# === Configuration ===
CONFIG = {
    'data_dir': 'Macine learing (bird deaseser)/final_poultry_dataset_10_classes',
    'output_dir': './cvt_poultry_results',
    'model_name': 'microsoft/cvt-13',
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
    print("POULTRY DISEASE CLASSIFICATION - CvT-13")
    print("=" * 60)

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

    # Load Processor
    processor = AutoImageProcessor.from_pretrained(CONFIG['model_name'])

    # Dataset Class
    class PoultryDataset(Dataset):
        def __init__(self, paths, labels, processor):
            self.paths = paths
            self.labels = labels
            self.processor = processor
            self.label2id = label2id

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            label = self.labels[idx]
            try:
                image = Image.open(path).convert('RGB')
                encoding = self.processor(image, return_tensors="pt")
                item = {k: v.squeeze() for k, v in encoding.items()}
                item['labels'] = torch.tensor(self.label2id[label])
                return item
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Return dummy
                image = Image.new('RGB', (224, 224))
                encoding = self.processor(image, return_tensors="pt")
                item = {k: v.squeeze() for k, v in encoding.items()}
                item['labels'] = torch.tensor(self.label2id[label])
                return item

    train_dataset = PoultryDataset(train_paths, train_labels, processor)
    val_dataset = PoultryDataset(val_paths, val_labels, processor)
    test_dataset = PoultryDataset(test_paths, test_labels, processor)

    # 2. Model Setup
    print(f"\nLoading {CONFIG['model_name']}...")
    model = CvtForImageClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(classes),
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
    print(classification_report(true_labels, preds, target_names=classes))

    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - CvT-13')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'))
    print("✅ Saved confusion matrix")

if __name__ == '__main__':
    main()
