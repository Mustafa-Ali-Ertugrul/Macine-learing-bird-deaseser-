"""Data transforms for training and validation."""
from torchvision import transforms
from typing import Callable


def get_train_transforms(image_size: int = 224) -> Callable:
    """
    Get training transforms with augmentation.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])


def get_val_transforms(image_size: int = 224) -> Callable:
    """
    Get validation/test transforms without augmentation.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
