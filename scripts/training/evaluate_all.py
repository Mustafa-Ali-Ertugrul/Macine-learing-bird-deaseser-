
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import classification_report
import numpy as np
import timm
import sys

# Import Simple CNN structure from the script if possible, or redefine
# Redefining to avoid import side effects if scripts are messy
def create_simple_model(num_classes):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

class PoultryDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data_frame = self.data_frame[self.data_frame['disease'] != 'unknown']
        self.classes = sorted(self.data_frame['disease'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        img_path = self.root_dir / self.data_frame.iloc[idx]['image_path']
        label = self.data_frame.iloc[idx]['disease']
        label_idx = self.class_to_idx[label]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label_idx

def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = PoultryDiseaseDataset(
        csv_file='../../data/metadata/poultry_labeled_12k.csv',
        root_dir='../../data/processed/poultry_dataset_512x512',
        transform=val_transform
    )
    
    # Split mostly to get the same validation set as training if possible
    # Note: If we just want to see performance, we can evaluate on a subset or full.
    # To match training logs, we should use the same seed.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Validation samples: {len(val_dataset)}")
    num_classes = len(dataset.classes)
    
    # Models to evaluate
    model_configs = [
        {"name": "ResNet18", "path": "best_poultry_disease_resnet.pth", "type": "resnet"},
        {"name": "Simple CNN", "path": "best_poultry_disease_simple.pth", "type": "simple"},
        {"name": "ViT", "path": "poultry_disease_vit.pth", "type": "vit"},
        {"name": "ResNeXt", "path": "best_poultry_disease_resnext.pth", "type": "resnext"},
        {"name": "ConvNeXt", "path": "best_poultry_disease_convnext.pth", "type": "convnext"},
        {"name": "CvT", "path": "best_poultry_disease_cvt.pth", "type": "cvt"},
    ]
    
    for config in model_configs:
        print("\n" + "="*50)
        print(f"📊 Evaluating {config['name']}...")
        print("="*50)
        
        if not os.path.exists(config['path']):
            print(f"❌ Model file not found: {config['path']}")
            continue
            
        try:
            # Initialize model
            if config['type'] == 'resnet':
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif config['type'] == 'simple':
                model = create_simple_model(num_classes)
            elif config['type'] == 'vit':
                model = models.vit_b_16(pretrained=False)
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            elif config['type'] == 'resnext':
                model = models.resnext50_32x4d(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif config['type'] == 'convnext':
                try:
                    model = models.convnext_tiny(pretrained=False)
                    # Adjust head
                    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
                except:
                    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
            elif config['type'] == 'cvt':
                try:
                    model = timm.create_model('cvt_13', pretrained=False, num_classes=num_classes)
                except:
                    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
            
            # Load weights
            # Use strict=False to be robust against minor layer name mismatches if any
            state_dict = torch.load(config['path'], map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            report = classification_report(all_labels, all_preds, target_names=dataset.classes)
            print(report)
            
        except Exception as e:
            print(f"❌ Error evaluating {config['name']}: {e}")

if __name__ == "__main__":
    evaluate_all()
