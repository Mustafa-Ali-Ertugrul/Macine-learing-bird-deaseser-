"""Tests for advanced augmentation."""
import pytest
import torch
import numpy as np

from src.data.advanced_augmentation import CutMix, MixUp, CutMixCriterion, MixUpCriterion


class TestCutMix:
    """Tests for CutMix augmentation."""
    
    @pytest.fixture
    def batch_data(self):
        """Create sample batch data."""
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 10, (8,))
        return images, labels
    
    def test_cutmix_creation(self):
        """Test CutMix initialization."""
        cutmix = CutMix(alpha=1.0, prob=0.5)
        
        assert cutmix.alpha == 1.0
        assert cutmix.prob == 0.5
    
    def test_cutmix_output_shape(self, batch_data):
        """Test CutMix preserves image shape."""
        images, labels = batch_data
        cutmix = CutMix(alpha=1.0, prob=1.0)  # Always apply
        
        mixed_images, labels1, labels2, lam = cutmix(images, labels)
        
        assert mixed_images.shape == images.shape
        assert labels1.shape == labels.shape
        assert labels2.shape == labels.shape
        assert 0 <= lam <= 1


class TestMixUp:
    """Tests for MixUp augmentation."""
    
    @pytest.fixture
    def batch_data(self):
        """Create sample batch data."""
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 10, (8,))
        return images, labels
    
    def test_mixup_creation(self):
        """Test MixUp initialization."""
        mixup = MixUp(alpha=0.4, prob=0.5)
        
        assert mixup.alpha == 0.4
        assert mixup.prob == 0.5
    
    def test_mixup_output_shape(self, batch_data):
        """Test MixUp preserves image shape."""
        images, labels = batch_data
        mixup = MixUp(alpha=0.4, prob=1.0)  # Always apply
        
        mixed_images, labels1, labels2, lam = mixup(images, labels)
        
        assert mixed_images.shape == images.shape
        assert labels1.shape == labels.shape
        assert labels2.shape == labels.shape
        assert 0 <= lam <= 1


class TestCutMixCriterion:
    """Tests for CutMix loss criterion."""
    
    def test_cutmix_criterion(self):
        """Test CutMix criterion computation."""
        import torch.nn as nn
        
        base_criterion = nn.CrossEntropyLoss()
        criterion = CutMixCriterion(base_criterion)
        
        outputs = torch.randn(8, 10)
        labels1 = torch.randint(0, 10, (8,))
        labels2 = torch.randint(0, 10, (8,))
        lam = 0.7
        
        loss = criterion(outputs, labels1, labels2, lam)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0


class TestMixUpCriterion:
    """Tests for MixUp loss criterion."""
    
    def test_mixup_criterion(self):
        """Test MixUp criterion computation."""
        import torch.nn as nn
        
        base_criterion = nn.CrossEntropyLoss()
        criterion = MixUpCriterion(base_criterion)
        
        outputs = torch.randn(8, 10)
        labels1 = torch.randint(0, 10, (8,))
        labels2 = torch.randint(0, 10, (8,))
        lam = 0.6
        
        loss = criterion(outputs, labels1, labels2, lam)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
