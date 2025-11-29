#!/usr/bin/env python3
"""
CvT-13 Evaluation Script
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm
from transformers import CvtForImageClassification, AutoImageProcessor

# === Configuration ===
CONFIG = {
    'data_dir': 'Macine learing (bird deaseser)/final_poultry_dataset_10_classes',
    'model_path': './cvt_poultry_results/final_model',
    'output_dir': './cvt_poultry_results',
    'batch_size': 32,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def evaluate():
    print("=" * 60)
    print("EVALUATION - CvT-13")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")

    # 1. Load Data
    data_dir = CONFIG['data_dir']
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(classes)} classes")

    image_paths = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        cls_images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.extend(cls_images)
        labels.extend([cls] * len(cls_images))

    label2id = {label: i for i, label in enumerate(classes)}

    # 2. Load Model & Processor
    print(f"Loading model from {CONFIG['model_path']}...")
    try:
        processor = AutoImageProcessor.from_pretrained(CONFIG['model_path'])
        model = CvtForImageClassification.from_pretrained(CONFIG['model_path'])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Did you run training yet?")
        return

    model = model.to(CONFIG['device'])
    model.eval()

    # Dataset
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
                image = Image.new('RGB', (224, 224))
                encoding = self.processor(image, return_tensors="pt")
                item = {k: v.squeeze() for k, v in encoding.items()}
                item['labels'] = torch.tensor(self.label2id[label])
                return item

    dataset = PoultryDataset(image_paths, labels, processor)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # 3. Inference
    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = {k: v.to(CONFIG['device']) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            
            outputs = model(**inputs)
            preds = outputs.logits.argmax(-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Overall Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Save report
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    report_path = os.path.join(CONFIG['output_dir'], 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=classes))
    print(f"📝 Report saved to {report_path}")

if __name__ == '__main__':
    evaluate()
