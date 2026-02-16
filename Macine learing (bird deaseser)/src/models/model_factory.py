"""Enhanced model factory with advanced architectures and better heads."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, List, Dict


class EnhancedClassifier(nn.Module):
    """Enhanced classifier head with dropout and optional MLP."""
    
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3, use_mlp: bool = False):
        super().__init__()
        
        if use_mlp:
            self.head = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.BatchNorm1d(in_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(in_features // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        
    def forward(self, x):
        return self.head(x)


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    use_enhanced_head: bool = True
) -> nn.Module:
    """
    Create model with specified architecture.
    
    Args:
        model_name: Name of model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        use_enhanced_head: Use enhanced classifier head
        
    Returns:
        PyTorch model
        
    Raises:
        ValueError: If model name is not supported
    """
    model_name = model_name.lower()
    classifier = EnhancedClassifier if use_enhanced_head else nn.Linear
    
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = classifier(model.fc.in_features, num_classes, dropout)
        
    elif model_name == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = classifier(model.fc.in_features, num_classes, dropout)
        
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = classifier(model.fc.in_features, num_classes, dropout)
        
    elif model_name == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        model.fc = classifier(model.fc.in_features, num_classes, dropout)
        
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b1':
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b2':
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = classifier(model.classifier[2].in_features, num_classes, dropout)
        
    elif model_name == 'convnext_small':
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
        model.classifier[2] = classifier(model.classifier[2].in_features, num_classes, dropout)
        
    elif model_name == 'convnext_base':
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
        model.classifier[2] = classifier(model.classifier[2].in_features, num_classes, dropout)
        
    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = classifier(model.heads.head.in_features, num_classes, dropout)
        
    elif model_name == 'swin_t':
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = classifier(model.head.in_features, num_classes, dropout)
        
    elif model_name == 'swin_s':
        weights = models.Swin_S_Weights.DEFAULT if pretrained else None
        model = models.swin_s(weights=weights)
        model.head = classifier(model.head.in_features, num_classes, dropout)
        
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: resnet18, resnet50, efficientnet_b0-b3, "
            f"convnext_tiny/small/base, vit_b_16, swin_t/s"
        )
    
    return model


def get_model_info(model_name: str) -> Dict:
    """Get model information (parameters, FLOPs estimate)."""
    info = {
        'resnet18': {'params': '11.7M', 'depth': 18},
        'resnet50': {'params': '25.6M', 'depth': 50},
        'resnet101': {'params': '44.5M', 'depth': 101},
        'efficientnet_b0': {'params': '5.3M', 'depth': 'B0'},
        'efficientnet_b2': {'params': '9.2M', 'depth': 'B2'},
        'efficientnet_b3': {'params': '12M', 'depth': 'B3'},
        'convnext_tiny': {'params': '28M', 'depth': 'tiny'},
        'convnext_small': {'params': '50M', 'depth': 'small'},
        'swin_t': {'params': '28M', 'depth': 'tiny'},
        'swin_s': {'params': '50M', 'depth': 'small'},
    }
    return info.get(model_name.lower(), {})


# Model recommendations for different scenarios
MODEL_RECOMMENDATIONS = {
    'fast': [
        ('efficientnet_b0', 'Fastest, good accuracy'),
        ('resnet18', 'Very fast, lower accuracy'),
    ],
    'balanced': [
        ('efficientnet_b2', 'Good balance of speed/accuracy'),
        ('resnet50', 'Classic choice'),
        ('convnext_tiny', 'Modern architecture'),
    ],
    'accurate': [
        ('efficientnet_b3', 'High accuracy'),
        ('convnext_small', 'Very high accuracy'),
        ('swin_s', 'State-of-the-art'),
    ],
}
