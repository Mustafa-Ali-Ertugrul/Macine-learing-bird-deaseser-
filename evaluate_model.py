import os
import torch
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
from tqdm import tqdm
import json

# Configuration
MODEL_PATH = "./vit_poultry_results/final_model"
DATA_DIR = "Macine learing (bird deaseser)/final_poultry_dataset_10_classes"
BATCH_SIZE = 16

class PoultryViTDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor, label2id):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.label2id = label2id
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = self.label2id[self.labels[idx]]
        return item

def evaluate():
    print(f"🔮 Loading model from {MODEL_PATH}...")
    
    try:
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Prepare dataset (Load all images for evaluation - or a subset)
    # For speed, let's use the validation/test split logic or just evaluate on a random subset
    # Here we will scan the directory again
    
    print(f"📁 Scanning dataset: {DATA_DIR}")
    image_paths = []
    labels = []
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}
    
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Let's take a 20% sample for quick evaluation if dataset is huge, 
        # but 5000 images is manageable. Let's use all of them to be accurate.
        image_paths.extend(files)
        labels.extend([cls] * len(files))
        
    print(f"📊 Evaluating on {len(image_paths)} images...")
    
    dataset = PoultryViTDataset(image_paths, labels, feature_extractor, label2id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    print("🚀 Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            batch_labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            preds = outputs.logits.argmax(-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print("\n" + "="*60)
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*60)
    
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    # Save report to file
    with open("evaluation_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=classes, digits=4))
        
    print("💾 Report saved to evaluation_report.txt")

if __name__ == "__main__":
    evaluate()
