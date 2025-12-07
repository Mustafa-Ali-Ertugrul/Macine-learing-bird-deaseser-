
import torch
import os

model_path = 'poultry_disease_vit.pth'

if os.path.exists(model_path):
    print(f"Inspecting {model_path}...")
    try:
        # Load on CPU
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check if it's a dict or model
        if isinstance(state_dict, dict):
            print("It's a state_dict or dict.")
            print(f"Keys: {list(state_dict.keys())[:20]}")
            
            # Check for classifier keys to guess classes
            for key in state_dict.keys():
                if ('classifier' in key or 'head' in key or 'fc' in key) and 'weight' in key:
                     if len(state_dict[key].shape) > 0:
                        print(f"Found potential output layer: {key} shape: {state_dict[key].shape}")
        else:
            print("It's a full model object.")
            
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Model not found: {model_path}")
