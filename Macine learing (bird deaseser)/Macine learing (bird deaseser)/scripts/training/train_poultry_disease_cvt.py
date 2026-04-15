
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import timm

class PoultryDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Filter out unknown images
        self.data_frame = self.data_frame[self.data_frame['disease'] != 'unknown']
        
        # Get unique classes
        self.classes = sorted(self.data_frame['disease'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Dataset classes: {self.classes}")
        print(f"Number of samples: {len(self.data_frame)}")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_path = self.root_dir / self.data_frame.iloc[idx]['image_path']
        label = self.data_frame.iloc[idx]['disease']
        
        # Convert label to index
        label_idx = self.class_to_idx[label]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx

def train_model():
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    dataset = PoultryDiseaseDataset(
        csv_file='../../data/metadata/final_dataset_10_classes.csv',
        root_dir='../../Macine learing (bird deaseser)/final_dataset_10_classes',
        transform=train_transform
    )
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply different transforms to validation set
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load Pre-trained CvT model
    print("Loading CvT-13...")
    # Using cvt_13 (Convolutional Vision Transformer)
    # create_model handles num_classes update automatically if specified
    try:
        model = timm.create_model('cvt_13', pretrained=True, num_classes=len(dataset.classes))
    except Exception as e:
        print(f"CvT-13 not found or error: {e}. Trying alternate vit...")
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=len(dataset.classes))
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Low learning rate for transformers
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = 10
    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_poultry_disease_cvt.pth')
            print(f'  🎉 Saved best model with validation accuracy: {val_acc:.2f}%')
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Print classification report
    print("\n📊 Classification Report:")
    target_names = dataset.classes
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Save final model
    torch.save(model.state_dict(), 'final_poultry_disease_cvt.pth')
    print("✅ Final model saved as 'final_poultry_disease_cvt.pth'")

if __name__ == "__main__":
    train_model()
