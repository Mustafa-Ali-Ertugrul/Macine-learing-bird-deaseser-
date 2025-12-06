import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from pathlib import Path
import numpy as np

class PoultryDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Filter out unknown images
        self.data_frame = self.data_frame[self.data_frame['disease'] != 'unknown']
        
        # Get unique classes
        unique_diseases = pd.Series(self.data_frame['disease']).drop_duplicates()
        self.classes = sorted(unique_diseases.tolist())
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

def create_simple_model(num_classes):
    """
    Create a simple CNN model for poultry disease classification
    """
    model = nn.Sequential(
        # First convolutional block
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        # Second convolutional block
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        # Third convolutional block
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        # Fourth convolutional block
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        # Flatten and fully connected layers
        nn.Flatten(),
        nn.Linear(256 * 14 * 14, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

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
        csv_file='../../data/metadata/poultry_labeled_12k.csv',
        root_dir='../../data/processed/poultry_dataset_512x512',
        transform=train_transform
    )
    
    # Check if we have enough data
    if len(dataset) < 10:
        print("❌ Not enough labeled data to train a model. Please label more images first.")
        return
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    # Create separate datasets with different transforms
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # Get item from original dataset
            image, label = self.dataset[idx]
            # Apply transform to image
            if self.transform:
                # Convert tensor back to PIL image for transform
                image = transforms.ToPILImage()(image)
                image = self.transform(image)
            return image, label
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create simple CNN model
    num_classes = len(dataset.classes)
    model = create_simple_model(num_classes)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_poultry_disease_simple.pth')
            print(f'  🎉 Saved best model with validation accuracy: {val_acc:.2f}%')
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'final_poultry_disease_simple.pth')
    print("✅ Final model saved as 'final_poultry_disease_simple.pth'")

def main():
    """
    Main function to train the poultry disease classification model
    """
    print("🐔 Poultry Disease Classification Model Trainer")
    print("=" * 50)
    
    # Check if CSV file exists
    csv_path = '../../data/metadata/poultry_labeled_12k.csv'
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please run the data preparation scripts first.")
        return
    
    # Train the model
    try:
        train_model()
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main()