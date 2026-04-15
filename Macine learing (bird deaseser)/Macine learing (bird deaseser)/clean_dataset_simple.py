
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import os
from tqdm import tqdm

# === Configuration ===
CSV_PATH = 'Macine learing (bird deaseser)/poultry_labeled.csv'
MODEL_PATH = 'models/best_poultry_disease_model.pth' # Relative to workspace root
IMAGE_BASE_DIR = 'data/processed/poultry_dataset_512x512'

# Ensure paths work from current working directory
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'Macine learing (bird deaseser)/models/best_poultry_disease_model.pth'
if not os.path.exists(MODEL_PATH):
     MODEL_PATH = 'Macine learing (bird deaseser)/best_poultry_disease_model.pth'

def create_simple_model(num_classes):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 14 * 14, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

def get_classes_and_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
        
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer num_classes from the last layer weight shape
    # Key '16.weight' has shape [out_features, in_features]
    if '16.weight' in state_dict:
        num_classes = state_dict['16.weight'].shape[0]
        print(f"Inferred num_classes: {num_classes}")
    else:
        # Fallback or error
        keys = list(state_dict.keys())
        last_key = keys[-2] # weight of last layer usually
        num_classes = state_dict[last_key].shape[0]
        print(f"Guessed num_classes from {last_key}: {num_classes}")

    # We need class names to map index to label.
    # Since we don't have the class mapping file saved with model, 
    # we assume they are sorted alphabetically from the training data directory.
    # We'll use the 10 classes if num_classes is 10.
    classes = []
    if num_classes == 10:
        classes = sorted(['Avian Influenza', 'Coccidiosis', 'Fowl Pox', 'Healthy', 
                          'Histomoniasis', 'Infectious Bronchitis', 'Infectious Bursal Disease', 
                          'Marek\'s Disease', 'Newcastle Disease', 'Salmonella'])
    else:
        # Try to find classes from data directory
        data_dir = 'Macine learing (bird deaseser)/final_dataset_10_classes'
        if os.path.exists(data_dir):
            classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            if len(classes) != num_classes:
                print(f"⚠️ Warning: Found {len(classes)} classes in dir but model has {num_classes}.")
                # This ensures we don't crash but predictions might be unmapped
                classes = [str(i) for i in range(num_classes)]
        else:
            classes = [str(i) for i in range(num_classes)]

    model = create_simple_model(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, classes

def predict(model, input_tensor, classes):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        idx = preds.item()
        if idx < len(classes):
            return classes[idx]
        return str(idx)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        model, classes = get_classes_and_model(MODEL_PATH, device)
        print(f"Classes map: {classes}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    rows_to_remove = []

    print(f"Processing {len(df)} images...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
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
            
            # Lowercase comparison
            # Note: 'unknown' in true_label will always mismatch if pred_label is a disease name
            if pred_label.lower() != true_label.lower():
                rows_to_remove.append(idx)
                
        except Exception as e:
            print(f"Error: {e}")

    print(f"\nFound {len(rows_to_remove)} mismatches to remove.")
    
    df_cleaned = df.drop(rows_to_remove)
    output_path = CSV_PATH.replace('.csv', '_cleaned.csv')
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned CSV to: {output_path}")

if __name__ == "__main__":
    main()
