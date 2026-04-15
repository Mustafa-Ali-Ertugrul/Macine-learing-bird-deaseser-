"""Tests for data module."""
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import os

from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.dataset import PoultryDiseaseDataset


class TestTransforms:
    """Tests for data transforms."""
    
    def test_train_transforms_creation(self):
        """Test training transforms creation."""
        transform = get_train_transforms(image_size=224)
        
        assert transform is not None
        
        # Test with dummy image
        img = Image.new('RGB', (300, 300), color='red')
        output = transform(img)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([3, 224, 224])
    
    def test_val_transforms_creation(self):
        """Test validation transforms creation."""
        transform = get_val_transforms(image_size=224)
        
        assert transform is not None
        
        # Test with dummy image
        img = Image.new('RGB', (300, 300), color='blue')
        output = transform(img)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([3, 224, 224])
    
    def test_different_image_sizes(self):
        """Test transforms with different target sizes."""
        for size in [128, 224, 256, 299]:
            transform = get_train_transforms(image_size=size)
            img = Image.new('RGB', (400, 400), color='green')
            output = transform(img)
            
            assert output.shape == torch.Size([3, size, size])


class TestDataset:
    """Tests for PoultryDiseaseDataset."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy dataset files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy images
            image_paths = []
            labels = []
            
            for i in range(5):
                img_path = Path(tmpdir) / f"image_{i}.jpg"
                img = Image.new('RGB', (300, 300), color=(i*50, i*50, i*50))
                img.save(img_path)
                image_paths.append(str(img_path))
                labels.append(f"class_{i % 2}")
            
            class_to_idx = {'class_0': 0, 'class_1': 1}
            
            yield image_paths, labels, class_to_idx
    
    def test_dataset_creation(self, dummy_data):
        """Test dataset creation."""
        image_paths, labels, class_to_idx = dummy_data
        transform = get_val_transforms(224)
        
        dataset = PoultryDiseaseDataset(
            image_paths=image_paths,
            labels=labels,
            class_to_idx=class_to_idx,
            transform=transform
        )
        
        assert len(dataset) == 5
        assert dataset.num_classes == 2
    
    def test_dataset_getitem(self, dummy_data):
        """Test dataset item retrieval."""
        image_paths, labels, class_to_idx = dummy_data
        transform = get_val_transforms(224)
        
        dataset = PoultryDiseaseDataset(
            image_paths=image_paths,
            labels=labels,
            class_to_idx=class_to_idx,
            transform=transform
        )
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert image.shape == torch.Size([3, 224, 224])
    
    def test_dataset_length(self, dummy_data):
        """Test dataset length."""
        image_paths, labels, class_to_idx = dummy_data
        
        dataset = PoultryDiseaseDataset(
            image_paths=image_paths,
            labels=labels,
            class_to_idx=class_to_idx,
            transform=None
        )
        
        assert len(dataset) == len(image_paths)
    
    def test_dataset_with_invalid_image(self, dummy_data):
        """Test dataset handling of invalid images."""
        image_paths, labels, class_to_idx = dummy_data
        
        # Add non-existent image path
        image_paths.append("/nonexistent/image.jpg")
        labels.append("class_0")
        
        transform = get_val_transforms(224)
        dataset = PoultryDiseaseDataset(
            image_paths=image_paths,
            labels=labels,
            class_to_idx=class_to_idx,
            transform=transform
        )
        
        # Should still be created, but may return None for invalid images
        assert len(dataset) == 6


class TestDataUtils:
    """Tests for data utilities."""
    
    def test_image_loading(self):
        """Test image loading."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create test image
            img = Image.new('RGB', (300, 300), color='purple')
            img.save(tmp.name)
            
            # Load image
            loaded = Image.open(tmp.name).convert('RGB')
            
            assert loaded.size == (300, 300)
            assert loaded.mode == 'RGB'
            
            # Cleanup
            os.unlink(tmp.name)
