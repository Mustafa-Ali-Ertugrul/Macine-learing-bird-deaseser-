"""Custom dataset for poultry disease classification."""
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Callable


class PoultryDiseaseDataset(Dataset):
    """Custom Dataset for Poultry Disease Classification."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of label indices
            class_to_idx: Mapping from class name to index
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
