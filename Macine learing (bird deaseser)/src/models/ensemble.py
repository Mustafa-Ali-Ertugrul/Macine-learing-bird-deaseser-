"""Model ensemble for improved predictions."""
import torch
import torch.nn as nn
from typing import List, Dict
from ..models.model_factory import get_model


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models with voting/averaging."""
    
    def __init__(
        self,
        models: List[nn.Module],
        method: str = 'average'  # 'average' or 'voting'
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == 'average':
            outputs = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    out = model(x)
                    outputs.append(out)
            return torch.stack(outputs).mean(dim=0)
        else:
            votes = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    out = model(x)
                    votes.append(out.argmax(dim=1))
            votes = torch.stack(votes)
            result = torch.mode(votes, dim=0).values
            return result


def create_ensemble(
    model_names: List[str],
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = 'cuda'
) -> ModelEnsemble:
    """
    Create ensemble of multiple models.
    
    Args:
        model_names: List of model architecture names
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        device: Device to load models on
        
    Returns:
        ModelEnsemble instance
    """
    models = []
    for name in model_names:
        print(f"Loading {name}...")
        model = get_model(
            model_name=name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
        model = model.to(device)
        models.append(model)
    
    return ModelEnsemble(models, method='average')


def load_ensemble_checkpoints(
    checkpoint_paths: List[str],
    model_names: List[str],
    num_classes: int,
    device: str = 'cuda'
) -> ModelEnsemble:
    """
    Load ensemble from saved checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        model_names: List of model architecture names
        num_classes: Number of output classes
        device: Device to load models on
        
    Returns:
        ModelEnsemble instance with loaded weights
    """
    models = []
    for name, path in zip(model_names, checkpoint_paths):
        print(f"Loading {name} from {path}...")
        model = get_model(
            model_name=name,
            num_classes=num_classes,
            pretrained=False
        )
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        models.append(model)
    
    return ModelEnsemble(models, method='average')


def get_ensemble_predictions(
    ensemble: ModelEnsemble,
    input_tensor: torch.Tensor,
    num_classes: int
) -> Dict:
    """
    Get detailed ensemble predictions.
    
    Args:
        ensemble: ModelEnsemble instance
        input_tensor: Input images [B, C, H, W]
        num_classes: Number of classes
        
    Returns:
        Dictionary with predictions and confidence
    """
    ensemble.eval()
    
    all_probs = []
    with torch.no_grad():
        for model in ensemble.models:
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs)
    
    stacked_probs = torch.stack(all_probs)
    mean_probs = stacked_probs.mean(dim=0)
    std_probs = stacked_probs.std(dim=0)
    
    predictions = mean_probs.argmax(dim=1)
    confidence = mean_probs.max(dim=1).values
    
    return {
        'predictions': predictions,
        'confidence': confidence,
        'mean_probs': mean_probs,
        'std_probs': std_probs,
        'agreement': 1 - std_probs.mean(dim=1)
    }
