"""Test-Time Augmentation (TTA) for improved inference."""
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List
import numpy as np


class TTAAugmentation:
    """Test-Time Augmentation - apply augmentations during inference."""
    
    def __init__(self, num_augmentations: int = 5):
        self.num_augmentations = num_augmentations
        self.transforms = self._create_transforms()
    
    def _create_transforms(self):
        """Create list of augmentation transforms."""
        tta_transforms = [
            lambda img: img,  # Original
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return tta_transforms[:self.num_augmentations]
    
    def __call__(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Apply TTA transforms to images."""
        augmented = []
        for transform in self.transforms:
            augmented.append(transform(images))
        return augmented


def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    num_augmentations: int = 5
) -> torch.Tensor:
    """
    Make predictions with TTA.
    
    Args:
        model: Trained model
        images: Input images [B, C, H, W]
        device: Device (cuda/cpu)
        num_augmentations: Number of augmentations to apply
        
    Returns:
        Averaged predictions
    """
    model.eval()
    
    tta_aug = TTAAugmentation(num_augmentations)
    all_predictions = []
    
    with torch.no_grad():
        augmented_images = tta_aug(images)
        
        for aug_images in augmented_images:
            aug_images = aug_images.to(device)
            outputs = model(aug_images)
            probs = torch.softmax(outputs, dim=1)
            all_predictions.append(probs)
    
    # Average predictions
    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    
    return avg_predictions


def tta_transform_inference(image, model, device, num_aug=5):
    """TTA inference for single image."""
    model.eval()
    
    tta_transforms = [
        lambda x: x,
        transforms.functional.hflip,
        transforms.functional.vflip,
        lambda x: transforms.functional.rotate(x, 15),
        lambda x: transforms.functional.rotate(x, -15),
    ]
    
    predictions = []
    
    with torch.no_grad():
        for i in range(min(num_aug, len(tta_transforms))):
            aug_img = tta_transforms[i](image).unsqueeze(0).to(device)
            output = model(aug_img)
            pred = torch.softmax(output, dim=1)
            predictions.append(pred)
    
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred
