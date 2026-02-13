"""Advanced augmentation techniques: CutMix and MixUp."""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class CutMix:
    """
    CutMix augmentation: Cut and paste patches between images.
    
    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            
        Returns:
            Mixed images, labels, labels of second image, lambda value
        """
        if np.random.rand() > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # Get image dimensions
        _, _, h, w = images.shape
        
        # Calculate bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return images, labels, labels[rand_index], lam


class MixUp:
    """
    MixUp augmentation: Blend two images with alpha blending.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization"
    """
    
    def __init__(self, alpha: float = 0.4, prob: float = 0.5):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying MixUp
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            
        Returns:
            Mixed images, labels, labels of second image, lambda value
        """
        if np.random.rand() > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[rand_index]
        
        return mixed_images, labels, labels[rand_index], lam


class CutMixCriterion:
    """Criterion for CutMix loss calculation."""
    
    def __init__(self, criterion: nn.Module):
        """
        Initialize CutMix criterion wrapper.
        
        Args:
            criterion: Base loss criterion
        """
        self.criterion = criterion
    
    def __call__(
        self,
        outputs: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Compute CutMix loss.
        
        Args:
            outputs: Model outputs
            labels1: First set of labels
            labels2: Second set of labels
            lam: Lambda mixing parameter
            
        Returns:
            Mixed loss
        """
        return lam * self.criterion(outputs, labels1) + (1 - lam) * self.criterion(outputs, labels2)


class MixUpCriterion:
    """Criterion for MixUp loss calculation."""
    
    def __init__(self, criterion: nn.Module):
        """
        Initialize MixUp criterion wrapper.
        
        Args:
            criterion: Base loss criterion
        """
        self.criterion = criterion
    
    def __call__(
        self,
        outputs: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Compute MixUp loss.
        
        Args:
            outputs: Model outputs
            labels1: First set of labels
            labels2: Second set of labels
            lam: Lambda mixing parameter
            
        Returns:
            Mixed loss
        """
        return lam * self.criterion(outputs, labels1) + (1 - lam) * self.criterion(outputs, labels2)


class AutoAugment:
    """
    AutoAugment for medical images (simplified version).
    Applies a sequence of augmentation policies.
    """
    
    def __init__(self, policies: Optional[list] = None):
        """
        Initialize AutoAugment.
        
        Args:
            policies: List of augmentation policies
        """
        self.policies = policies or self._default_policies()
    
    def _default_policies(self):
        """Get default augmentation policies for medical images."""
        return [
            # Policy 1: Rotation + Color jitter
            [('rotate', 0.5, 15), ('color', 0.5, 0.2)],
            # Policy 2: Shear + Sharpness
            [('shear', 0.4, 10), ('sharpness', 0.3, 0.3)],
            # Policy 3: Translate + Contrast
            [('translate', 0.6, 0.1), ('contrast', 0.5, 0.2)],
        ]
    
    def __call__(self, img):
        """Apply random policy to image."""
        # Select random policy
        policy = self.policies[np.random.randint(len(self.policies))]
        
        # Apply transformations in policy
        for op, prob, mag in policy:
            if np.random.rand() < prob:
                img = self._apply_op(img, op, mag)
        
        return img
    
    def _apply_op(self, img, op, mag):
        """Apply single operation."""
        from PIL import Image
        import torchvision.transforms as transforms
        
        if op == 'rotate':
            return img.rotate(mag)
        elif op == 'shear':
            return img.transform(img.size, Image.Transform.AFFINE, (1, mag/100, 0, 0, 1, 0))
        elif op == 'translate':
            return img.transform(img.size, Image.Transform.AFFINE, (1, 0, mag*img.size[0], 0, 1, 0))
        elif op == 'color':
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(1 + mag)
        elif op == 'sharpness':
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(1 + mag)
        elif op == 'contrast':
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1 + mag)
        
        return img


# Import required for AutoAugment
try:
    from PIL import ImageEnhance
except ImportError:
    pass
