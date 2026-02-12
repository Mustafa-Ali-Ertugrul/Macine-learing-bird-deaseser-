"""Model factory for creating different architectures."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """
    Create model with specified architecture.
    
    Args:
        model_name: Name of model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate (if applicable)
        
    Returns:
        PyTorch model
        
    Raises:
        ValueError: If model name is not supported
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b2':
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: resnet18, resnet50, efficientnet_b0, "
            f"efficientnet_b2, convnext_tiny"
        )
    
    return model
