#!/usr/bin/env python3
"""
ResNeXt-50 Evaluation Script
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === Configuration ===
CONFIG = {
    'data_dir': 'Macine learing (bird deaseser)/final_poultry_dataset_10_classes',
    'model_path': './resnext_poultry_results/best_resnext.pth',
    'output_dir': './resnext_poultry_results',
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def evaluate():
    print("=" * 60)
    print("EVALUATION - ResNeXt-50")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")

    # 1. Load Data
    data_dir = CONFIG['data_dir']
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(classes)} classes")

    # Collect all images
    image_paths = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        cls_images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.extend(cls_images)
        labels.extend([cls] * len(cls_images))

    label2id = {label: i for i, label in enumerate(classes)}
    
    # Dataset
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

    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PoultryDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # 2. Load Model
    print(f"Loading model from {CONFIG['model_path']}...")
    model = models.resnext50_32x4d(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    except FileNotFoundError:
        print(f"❌ Model file not found at {CONFIG['model_path']}")
        print("Did you run training yet?")
        return

    model = model.to(CONFIG['device'])
    model.eval()

    # 3. Inference
    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(CONFIG['device'])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Overall Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Save report
    report_path = os.path.join(CONFIG['output_dir'], 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=classes))
    print(f"📝 Report saved to {report_path}")

if __name__ == '__main__':
    evaluate()
