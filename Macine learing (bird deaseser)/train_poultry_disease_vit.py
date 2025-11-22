
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import Path
import os

class PoultryDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted(self.data_frame['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.root_dir / self.data_frame.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.data_frame.iloc[idx]['class']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model():
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PoultryDiseaseDataset(
        csv_file='organized_poultry_dataset/train.csv',
        root_dir='organized_poultry_dataset',
        transform=train_transform
    )
    
    val_dataset = PoultryDiseaseDataset(
        csv_file='organized_poultry_dataset/val.csv',
        root_dir='organized_poultry_dataset',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load pre-trained Vision Transformer
    model = models.vit_b_16(pretrained=True)
    
    # Modify classifier for our number of classes
    num_classes = len(train_dataset.classes)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = 1
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'  Val Accuracy: {100 * correct / total:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'poultry_disease_vit.pth')
    print("✅ Model saved as 'poultry_disease_vit.pth'")

if __name__ == "__main__":
    train_model()
