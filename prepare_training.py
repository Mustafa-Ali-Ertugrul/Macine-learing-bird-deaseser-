import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_val_test_split(dataset_dir, csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/validation/test splits for the dataset
    """
    dataset_dir = Path(dataset_dir)
    csv_path = Path(csv_path)
    
    # Read the dataset CSV
    df = pd.read_csv(csv_path)
    
    # Split into train and temp (val + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=42, 
        stratify=df['class'] if 'class' in df.columns else None
    )
    
    # Split temp into val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=val_ratio_adjusted, 
        random_state=42, 
        stratify=temp_df['class'] if 'class' in temp_df.columns else None
    )
    
    # Save splits
    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    test_csv = dataset_dir / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"✅ Created dataset splits:")
    print(f"   Train: {len(train_df)} samples ({train_csv})")
    print(f"   Validation: {len(val_df)} samples ({val_csv})")
    print(f"   Test: {len(test_df)} samples ({test_csv})")
    
    # Print class distribution
    print(f"\n📊 Class distribution:")
    print(f"   Train set:")
    if 'class' in train_df.columns:
        print(train_df['class'].value_counts())
    
    return train_csv, val_csv, test_csv

def create_model_training_script():
    """
    Create a basic Vision Transformer training script
    """
    script_content = '''
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
'''
    
    script_path = Path("train_poultry_disease_vit.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ Created model training script: {script_path}")
    return script_path

def main():
    """
    Main function to prepare for model training
    """
    print("🐔 Preparing for Poultry Disease Model Training")
    print("=" * 50)
    
    dataset_dir = Path("organized_poultry_dataset")
    csv_path = dataset_dir / "dataset.csv"
    
    if not csv_path.exists():
        print("❌ Dataset CSV not found!")
        print("Please run the dataset organization scripts first.")
        return
    
    # Create train/val/test splits
    print("📊 Creating dataset splits...")
    train_csv, val_csv, test_csv = create_train_val_test_split(dataset_dir, csv_path)
    
    # Create training script
    print("🐍 Creating model training script...")
    train_script = create_model_training_script()
    
    print(f"\n🎉 Preparation complete!")
    print(f"📁 Dataset directory: {dataset_dir}")
    print(f"📄 Dataset CSV: {csv_path}")
    print(f"📄 Train CSV: {train_csv}")
    print(f"📄 Validation CSV: {val_csv}")
    print(f"📄 Test CSV: {test_csv}")
    print(f"🐍 Training script: {train_script}")
    print(f"\n🚀 To train your model, run:")
    print(f"   python {train_script}")

if __name__ == "__main__":
    main()