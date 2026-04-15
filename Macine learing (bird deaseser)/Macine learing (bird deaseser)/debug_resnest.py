
import timm
import torch

try:
    print("Attempting to create model...")
    model = timm.create_model('resnest50d', pretrained=True, num_classes=10)
    print("Model created successfully!")
except Exception as e:
    print(f"Error creating model: {e}")
    import traceback
    traceback.print_exc()
