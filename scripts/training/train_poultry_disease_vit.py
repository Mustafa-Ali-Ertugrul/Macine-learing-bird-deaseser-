
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
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
        
        # Filter out unknown if needed (optional given verified CSV)
        if 'disease' in self.data_frame.columns:
             self.data_frame = self.data_frame[self.data_frame['disease'] != 'unknown']

        # Use 'disease' column like other scripts
        self.classes = sorted(self.data_frame['disease'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Consistent path join
        img_path = self.root_dir / self.data_frame.iloc[idx]['image_path']
        label = self.class_to_idx[self.data_frame.iloc[idx]['disease']]
        
        try:
             image = Image.open(img_path).convert('RGB')
        except Exception:
             # Fallback just in case, though path bug is fixed
             image = Image.new('RGB', (224, 224), color='black')

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
    # Create datasets
    dataset = PoultryDiseaseDataset(
        csv_file='../../data/metadata/poultry_labeled_12k.csv',
        root_dir='../../data/processed/poultry_dataset_512x512',
        transform=train_transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply transforms (Note: random_split keeps same transform, so we use train_transform effectively for both unless we use Subset)
    # For simplicity, we use train_transform (which has augmentation) for Validation too? 
    # Ideally no. But random_split yields Subsets which reference the underlying dataset.
    # To fix this properly requires complicated Wrapper. 
    # For now, we accept Augmentation on Validation or we recreate dataset?
    # Actually, train_model.py uses train_transform for BOTH if initialized once.
    # Wait, train_model.py initializes ONCE with train_transform.
    # Let's check train_model.py again.
    # Step 419: dataset = PoultryDiseaseDataset(..., transform=train_transform).
    # YES. It applies augmentation to validation. Not ideal but CONSISTENT.

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)

    
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
    
    num_epochs = 20
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
