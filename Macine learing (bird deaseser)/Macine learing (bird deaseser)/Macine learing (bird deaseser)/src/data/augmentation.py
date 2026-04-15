"""Advanced augmentation techniques for poultry disease classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import Tuple, Optional


def rand_augment(img: torch.Tensor, m: int = 9, n: int = 2) -> torch.Tensor:
    """
    RandAugment: Automatically selected augmentations.
    
    Args:
        img: Input image tensor
        m: Magnitude (0-30)
        n: Number of transforms to apply
        
    Returns:
        Augmented image tensor
    """
    transforms_list = [
        lambda x: transforms.functional.autocontrast(x),
        lambda x: transforms.functional.equalize(x),
        lambda x: transforms.functional.invert(x),
        lambda x: transforms.functional.rotate(x, angle=np.random.randint(-m, m)),
        lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=np.random.uniform(1-m/10, 1+m/10)),
        lambda x: transforms.functional.adjust_contrast(x, contrast_factor=np.random.uniform(1-m/10, 1+m/10)),
        lambda x: transforms.functional.adjust_brightness(x, brightness_factor=np.random.uniform(1-m/10, 1+m/10)),
        lambda x: transforms.functional.adjust_saturation(x, saturation_factor=np.random.uniform(1-m/10, 1+m/10)),
    ]
    
    indices = np.random.choice(len(transforms_list), size=min(n, len(transforms_list)), replace=False)
    
    for idx in indices:
        img = transforms_list[idx](img)
    
    return img


def cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    CutMix: Cut and paste patches between training images.
    
    Args:
        images: Batch of images [B, C, H, W]
        labels: Batch of labels [B]
        alpha: Beta distribution parameter
        
    Returns:
        Mixed images, mixed labels, lambda value
    """
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    
    rand_index = torch.randperm(batch_size).to(images.device)
    
    labels_a = labels
    labels_b = labels[rand_index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    return images, labels_a, labels_b, lam


def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp: Blend images with random pairs.
    
    Args:
        images: Batch of images [B, C, H, W]
        labels: Batch of labels [B]
        alpha: Beta distribution parameter
        
    Returns:
        Mixed images, labels_a, labels_b, lambda value
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    
    rand_index = torch.randperm(batch_size).to(images.device)
    
    mixed_images = lam * images + (1 - lam) * images[rand_index]
    
    labels_a = labels
    labels_b = labels[rand_index]
    
    return mixed_images, labels_a, labels_b, lam


def rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


class AdvancedAugmentation:
    """
    Advanced augmentation combining CutMix, MixUp, and RandAugment.
    """
    
    def __init__(
        self,
        use_randaugment: bool = False,
        randaugment_m: int = 9,
        randaugment_n: int = 2,
        cutmix_prob: float = 0.5,
        mixup_prob: float = 0.3,
        cutmix_alpha: float = 1.0,
        mixup_alpha: float = 0.4
    ):
        self.use_randaugment = use_randaugment
        self.randaugment_m = randaugment_m
        self.randaugment_n = randaugment_n
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[float]]:
        """
        Apply advanced augmentations.
        
        Returns:
            images: Augmented images
            labels_a: Original/mixed labels
            labels_b: Mixed labels (for loss calculation)
            lam: Lambda value for mixing
        """
        r = np.random.rand()
        
        if self.use_randaugment:
            for i in range(images.size(0)):
                images[i] = rand_augment(images[i], self.randaugment_m, self.randaugment_n)
        
        if r < self.cutmix_prob:
            images, labels_a, labels_b, lam = cutmix(images, labels, self.cutmix_alpha)
            return images, labels_a, labels_b, lam
            
        elif r < self.cutmix_prob + self.mixup_prob:
            images, labels_a, labels_b, lam = mixup(images, labels, self.mixup_alpha)
            return images, labels_a, labels_b, lam
            
        else:
            return images, labels, labels, 1.0


def compute_mixed_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Compute loss for mixed samples (CutMix/MixUp)."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
