
import torch
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm
from collections import Counter

# === Configuration ===
DATA_DIR = 'Macine learing (bird deaseser)/final_dataset_10_classes'
REPORT_PATH = 'analysis_report.txt'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Pretrained ResNet50
    print("Loading ResNet50 (ImageNet)...")
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        categories = weights.meta["categories"]
    except Exception as e:
        print(f"Error loading V2 weights, trying default: {e}")
        model = models.resnet50(pretrained=True)
        # Fallback categories (simplified or placeholder)
        categories = [str(i) for i in range(1000)]

    model.to(device)
    model.eval()

    # 2. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Analyze
    report_lines = []
    
    for cls_name in sorted(os.listdir(DATA_DIR)):
        cls_path = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_path):
            continue
            
        print(f"\nAnalyzing {cls_name}...")
        report_lines.append(f"\n--- Class: {cls_name} ---")
        
        preds = []
        image_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(image_files, desc=cls_name[:20]):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    prob, cat_id = torch.topk(probabilities, 1)
                    
                    cat_id = cat_id.item()
                    label = categories[cat_id] if cat_id < len(categories) else str(cat_id)
                    preds.append(label)
                    
            except Exception as e:
                preds.append("Error")
        
        # Stats
        cnt = Counter(preds)
        print("Top 10 predictions:")
        for label, count in cnt.most_common(10):
            print(f"  {label}: {count}")
            report_lines.append(f"{label}: {count}")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
