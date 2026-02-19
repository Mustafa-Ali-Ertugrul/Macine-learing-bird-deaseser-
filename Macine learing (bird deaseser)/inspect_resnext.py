
import torch
import os

model_path = 'results/resnext_poultry_results/best_resnext.pth'

if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        keys = list(state_dict.keys())
        print(f"Keys: {keys[:5]} ... {keys[-5:]}")
        # Check last layer
        last_key = keys[-1] # Usually bias of fc
        print(f"Last key: {last_key} shape: {state_dict[last_key].shape}")
        last_weight_key = keys[-2]
        print(f"Weight key: {last_weight_key} shape: {state_dict[last_weight_key].shape}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Model not found: {model_path}")
