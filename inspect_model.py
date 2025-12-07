
import torch
import os

model_path = 'models/best_poultry_disease_model.pth'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    # Try the inner one
    model_path = 'Macine learing (bird deaseser)/best_poultry_disease_model.pth'

if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"Loaded state dict from {model_path}")
        print(f"Keys: {list(state_dict.keys())[:20]}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Model still not found")
