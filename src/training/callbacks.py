"""Training callbacks."""
from typing import Optional, Callable
import torch
import torch.nn as nn
from pathlib import Path


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
                return True
        
        return False


class ModelCheckpoint:
    """Save model checkpoints."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_acc',
        mode: str = 'max',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'max' or 'min'
            save_best_only: Only save when metric improves
            verbose: Whether to print messages
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_score = None
        
        if mode == 'max':
            self.is_better = lambda score, best: score > best
        else:
            self.is_better = lambda score, best: score < best
    
    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict
    ):
        """
        Save checkpoint if conditions are met.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            return
        
        if self.best_score is None or self.is_better(current_score, self.best_score):
            self.best_score = current_score
            
            if self.save_best_only:
                self._save_model(model, epoch, current_score)
        elif not self.save_best_only:
            self._save_model(model, epoch, current_score)
    
    def _save_model(self, model: nn.Module, epoch: int, score: float):
        """Save model to disk."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.filepath)
        
        if self.verbose:
            print(f"Model saved to {self.filepath} (epoch={epoch}, {self.monitor}={score:.4f})")


class LearningRateMonitor:
    """Monitor and log learning rate."""
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize LR monitor.
        
        Args:
            log_every_n_epochs: Log every N epochs
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.lr_history = []
    
    def __call__(self, optimizer, epoch: int):
        """
        Log learning rate.
        
        Args:
            optimizer: Optimizer instance
            epoch: Current epoch
        """
        if epoch % self.log_every_n_epochs == 0:
            lr = optimizer.param_groups[0]['lr']
            self.lr_history.append(lr)
            print(f"Epoch {epoch}: Learning rate = {lr:.6f}")
