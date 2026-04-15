
import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
from tqdm import tqdm

# === Configuration ===
DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
QUARANTINE_DIR = 'Macine learing (bird deaseser)/quarantine'

# Banned ImageNet Classes (from analysis)
# These are classes that appeared frequently in the analysis but are clearly not poultry diseases.
BANNED_KEYWORDS = [
    'nail', 'shovel', 'doormat', 'chain', 'hook', 'screw',
    'book_jacket', 'comic_book', 'web_site', 'envelope', 'packet', 'menu',
    'mask', 'safety_pin', 'stole', 'feather_boa', 'wig', 'lab_coat',
    'ping-pong_ball', 'projectile', 'balloon',
    'soap_dispenser', 'lotion', 'cream',
    'monitor', 'screen', 'television',
    'butcher_shop', 'grocery_store', 'restaurant',
    'dough', 'burrito', 'cheeseburger', 'hot_pot', 'meat_loaf', 'ice_cream', 'plate',
    'beaker', 'flask', 'measuring_cup'
]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(QUARANTINE_DIR):
        os.makedirs(QUARANTINE_DIR)
        
    # 1. Load Pretrained ResNet50
    print("Loading ResNet50 for cleaning...")
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        categories = weights.meta["categories"]
    except:
        model = models.resnet50(pretrained=True)
        categories = [str(i) for i in range(1000)] # Fallback

    model.to(device)
    model.eval()

    # 2. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Clean
    moved_total = 0
    
    for cls_name in sorted(os.listdir(DATA_DIR)):
        cls_path = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_path):
            continue
            
        print(f"\nCleaning {cls_name}...")
        image_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(image_files, desc="Scanning"):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prob, cat_id = torch.topk(output, 1)
                    label = categories[cat_id.item()]
                
                # Check banning conditions
                is_banned = False
                label_norm = label.replace(' ', '_').lower()
                
                for ban in BANNED_KEYWORDS:
                    if ban in label_norm:
                        is_banned = True
                        break
                
                if is_banned:
                    # Move to quarantine
                    dst_dir = os.path.join(QUARANTINE_DIR, cls_name)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    
                    dst_path = os.path.join(dst_dir, img_name)
                    shutil.move(img_path, dst_path)
                    moved_total += 1
                    # print(f"  Moved {img_name} (Classified as: {label})")
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"\n✅ Cleaning Complete. Moved {moved_total} images to {QUARANTINE_DIR}")

if __name__ == "__main__":
    main()
