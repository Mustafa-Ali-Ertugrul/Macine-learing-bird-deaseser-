"""Tests for model factory."""
import pytest
import torch
from src.models.model_factory import get_model


class TestModelFactory:
    """Tests for model factory."""
    
    @pytest.mark.parametrize("model_name", [
        "resnet18",
        "resnet50",
        "efficientnet_b0"
    ])
    def test_model_creation(self, model_name):
        """Test model creation for different architectures."""
        num_classes = 10
        model = get_model(model_name, num_classes, pretrained=False)
        
        assert model is not None
        
        # Test forward pass
        batch = torch.randn(2, 3, 224, 224)
        output = model(batch)
        
        assert output.shape == (2, num_classes)
    
    def test_invalid_model(self):
        """Test invalid model name."""
        with pytest.raises(ValueError):
            get_model("invalid_model", 10, pretrained=False)
    
    def test_pretrained_model(self):
        """Test pretrained model loading."""
        model = get_model("resnet18", 10, pretrained=True)
        assert model is not None
