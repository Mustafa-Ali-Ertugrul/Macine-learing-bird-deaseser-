
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
from tqdm import tqdm

# === Configuration ===
CSV_PATH = 'Macine learing (bird deaseser)/poultry_labeled.csv'
MODEL_PATH = 'results/resnext_poultry_results/best_resnext.pth' 
IMAGE_BASE_DIR = 'data/processed/poultry_dataset_512x512' 
DATA_DIR_FOR_CLASSES = 'Macine learing (bird deaseser)/final_dataset_10_classes'

def get_classes(data_dir):
    if not os.path.exists(data_dir):
        print(f"⚠️ Warning: Data dir {data_dir} not found. Using default 10 classes.")
        return sorted(['Avian Influenza', 'Coccidiosis', 'Fowl Pox', 'Healthy', 
                       'Histomoniasis', 'Infectious Bronchitis', 'Infectious Bursal Disease', 
                       'Marek\'s Disease', 'Newcastle Disease', 'Salmonella'])
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def load_model(model_path, num_classes, device):
    print(f"🔮 Loading ResNeXt-50 model from {model_path}...")
    model = models.resnext50_32x4d(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model = model.to(device)
    model.eval()
    return model

def predict(model, input_tensor, classes):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        idx = preds.item()
        return classes[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Get Classes
    classes = get_classes(DATA_DIR_FOR_CLASSES)
    print(f"Classes: {classes}")

    # 2. Load Model
    try:
        model = load_model(MODEL_PATH, len(classes), device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Process CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV with {len(df)} rows.")

    rows_to_remove = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
        rel_path = row['image_path']
        rel_path = rel_path.replace('\\', '/')
        img_path = os.path.join(IMAGE_BASE_DIR, rel_path)
        
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            pred_label = predict(model, input_tensor, classes)
            
            true_label = str(row['disease'])
            
            # If true label is 'unknown', we treat it as mismatch if user wants to remove unknown.
            # If 'healthy' (lowercase) vs 'Healthy' (Title Case), we should normalize.
            
            if pred_label.lower() != true_label.lower():
                # print(f"Mismatch: {img_path} (True: {true_label}, Pred: {pred_label})")
                rows_to_remove.append(idx)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\nFound {len(rows_to_remove)} rows to remove out of {len(df)}.")
    
    # 5. Save Cleaned CSV
    # We save as 'poultry_labeled_cleaned.csv' as requested in user snippet
    output_path = CSV_PATH.replace('poultry_labeled.csv', 'poultry_labeled_cleaned.csv')
    df_cleaned = df.drop(rows_to_remove)
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned CSV to: {output_path}")

if __name__ == "__main__":
    main()
