
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
from tqdm import tqdm

# === Configuration ===
CSV_PATH = 'Macine learing (bird deaseser)/poultry_labeled_12k.csv'
MODEL_PATH = 'results/resnext_poultry_results/best_resnext.pth' # Relative to workspace
IMAGE_BASE_DIR = 'data/processed/poultry_dataset_512x512' 
DATA_DIR_FOR_CLASSES = 'Macine learing (bird deaseser)/final_dataset_10_classes'

# Label Mapping (CSV Label -> Model Class)
LABEL_MAP = {
    'coccidiosis': 'Coccidiosis',
    'healthy': 'Healthy', 
    'ncd': 'Newcastle Disease',
    'pcrcocci': 'Coccidiosis',
    'pcrhealthy': 'Healthy',
    'pcrncd': 'Newcastle Disease',
    'pcrsalmo': 'Salmonella',
    'salmonella': 'Salmonella',
    'unknown': 'unknown' # Special handler
}

# Model Classes (Reference)
# ['Avian Influenza...', 'Coccidiosis', 'Fowl Pox', 'Healthy', 'Histomoniasis', 
#  'Infectious Bronchitis', 'Infectious Bursal Disease', 'Marek\'s Disease', 'Newcastle Disease', 'Salmonella']

def get_classes(data_dir):
    if not os.path.exists(data_dir):
        print(f"⚠️ Warning: Data dir {data_dir} not found. Using default classes.")
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
    # Simplify class names for matching if needed (e.g. remove extra text)
    # But Model was trained on these specific folder names, so we return them as is for prediction.
    # We need to ensure we map TO these names.
    print(f"Model Classes: {classes}")

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
    
    # Helper to check partial match for labels
    def match_label(pred, target_mapped):
        if target_mapped == 'unknown':
            return False # Always mismatch
        
        # Normalize strings
        p = pred.lower()
        t = target_mapped.lower()
        
        # Exact match
        if p == t: return True
        
        # Partial match (e.g. 'Avian Influenza...' vs 'Avian Influenza')
        if t in p or p in t:
            return True
            
        return False

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
        rel_path = row['image_path']
        if pd.isna(rel_path):
            rows_to_remove.append(idx)
            continue
            
        rel_path = rel_path.replace('\\', '/')
        img_path = os.path.join(IMAGE_BASE_DIR, rel_path)
        
        if not os.path.exists(img_path):
            # If image missing, remove row?
            # print(f"Missing: {img_path}")
            rows_to_remove.append(idx)
            continue
            
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            pred_label = predict(model, input_tensor, classes)
            
            raw_label = str(row['disease']).lower()
            mapped_label = LABEL_MAP.get(raw_label, 'unknown')
            
            if not match_label(pred_label, mapped_label):
                # print(f"Mismatch: {raw_label} -> {mapped_label} != {pred_label}")
                rows_to_remove.append(idx)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            rows_to_remove.append(idx)

    print(f"\nFound {len(rows_to_remove)} rows to remove out of {len(df)}.")
    
    # 5. Save Cleaned CSV
    # Save as poultry_labeled_12k_cleaned.csv
    output_path = CSV_PATH.replace('.csv', '_cleaned.csv')
    df_cleaned = df.drop(rows_to_remove)
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned CSV to: {output_path}")
    
    # Validation stats
    print("\nOriginal Distribution:")
    print(df['disease'].value_counts().head())
    print("\nCleaned Distribution:")
    print(df_cleaned['disease'].value_counts().head())

if __name__ == "__main__":
    main()
