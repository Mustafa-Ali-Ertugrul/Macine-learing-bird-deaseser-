"""Tests for training module."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import Trainer, compute_metrics, format_classification_report
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.models.model_factory import get_model


class TestTrainer:
    """Tests for Trainer class."""
    
    @pytest.fixture
    def config(self):
        """Sample config for testing."""
        return {
            'model': {
                'architecture': 'resnet18',
                'pretrained': False,
                'dropout': 0.3
            },
            'training': {
                'num_epochs': 2,
                'batch_size': 4,
                'num_workers': 0,
                'optimizer': {
                    'type': 'adam',
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
                'loss': {
                    'use_class_weights': False,
                    'label_smoothing': 0.0
                },
                'early_stopping': {
                    'enabled': False,
                    'patience': 5
                },
                'mixed_precision': False
            },
            'data': {
                'image_size': 224,
                'num_classes': 5
            },
            'logging': {
                'log_dir': 'logs',
                'save_model_dir': 'test_models'
            }
        }
    
    @pytest.fixture
    def dataloaders(self):
        """Create dummy dataloaders for testing."""
        # Create dummy data
        train_data = torch.randn(20, 3, 224, 224)
        train_labels = torch.randint(0, 5, (20,))
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        val_data = torch.randn(10, 3, 224, 224)
        val_labels = torch.randint(0, 5, (10,))
        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        return train_loader, val_loader
    
    def test_trainer_creation(self, config, dataloaders):
        """Test trainer initialization."""
        train_loader, val_loader = dataloaders
        model = get_model('resnet18', num_classes=5, pretrained=False)
        
        trainer = Trainer(model, train_loader, val_loader, config)
        
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
    
    def test_train_epoch(self, config, dataloaders):
        """Test single epoch training."""
        train_loader, val_loader = dataloaders
        model = get_model('resnet18', num_classes=5, pretrained=False)
        
        trainer = Trainer(model, train_loader, val_loader, config)
        
        loss, acc = trainer.train_epoch()
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    
    def test_validate(self, config, dataloaders):
        """Test validation."""
        train_loader, val_loader = dataloaders
        model = get_model('resnet18', num_classes=5, pretrained=False)
        
        trainer = Trainer(model, train_loader, val_loader, config)
        
        loss, acc, preds, labels = trainer.validate()
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(preds, list)
        assert isinstance(labels, list)
        assert 0 <= acc <= 100


class TestMetrics:
    """Tests for metrics computation."""
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        class_names = ['class_0', 'class_1', 'class_2']
        
        metrics = compute_metrics(y_true, y_pred, class_names)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'per_class' in metrics
        assert 'confusion_matrix' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
    
    def test_format_classification_report(self):
        """Test classification report formatting."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        class_names = ['class_0', 'class_1', 'class_2']
        
        report = format_classification_report(y_true, y_pred, class_names)
        
        assert isinstance(report, str)
        assert 'class_0' in report
        assert 'class_1' in report


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""
    
    def test_early_stopping_creation(self):
        """Test early stopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.01, mode='max')
        
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == 'max'
        assert not es.early_stop
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        es = EarlyStopping(patience=2, mode='max')
        
        assert not es(0.5)  # First call
        assert not es(0.6)  # Improved
        assert not es(0.6)  # No improvement, counter=1
        assert not es(0.6)  # No improvement, counter=2
        assert es(0.6)      # Should trigger early stopping
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode."""
        es = EarlyStopping(patience=2, mode='min')
        
        assert not es(0.5)  # First call
        assert not es(0.4)  # Improved (lower is better)
        assert not es(0.4)  # No improvement
        assert not es(0.4)  # No improvement
        assert es(0.4)      # Should trigger
