"""Training pipeline for poultry disease classification."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List, Callable
import time
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Initialize training components
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # TensorBoard
        self.writer = None
        if config.get('logging', {}).get('tensorboard_dir'):
            log_dir = Path(config['logging']['tensorboard_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        loss_config = self.config.get('training', {}).get('loss', {})
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        return nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config.get('training', {}).get('optimizer', {})
        opt_type = opt_config.get('type', 'adam').lower()
        lr = opt_config.get('lr', 0.001)
        weight_decay = opt_config.get('weight_decay', 0.0)
        
        # Only train unfrozen parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if opt_type == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_type == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_type == 'sgd':
            momentum = opt_config.get('momentum', 0.9)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 3),
                verbose=True
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            num_epochs = self.config.get('training', {}).get('num_epochs', 50)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs
            )
        return None
    
    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.get('training', {}).get('mixed_precision', False):
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> tuple[float, float, list, list]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, num_epochs: int) -> Dict:
        """
        Train the model for specified epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, _, _ = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar(
                    'Learning_rate',
                    self.optimizer.param_groups[0]['lr'],
                    epoch
                )
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch [{self.current_epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                self.save_checkpoint(is_best=True)
                logger.info(f"New best validation accuracy: {val_acc:.2f}%")
            else:
                self.epochs_no_improve += 1
            
            early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
            if early_stopping_config.get('enabled', False):
                patience = early_stopping_config.get('patience', 10)
                if self.epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, is_best: bool = False, filepath: Optional[str] = None):
        """Save model checkpoint."""
        if filepath is None:
            save_dir = Path(self.config.get('logging', {}).get('save_model_dir', 'models'))
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if is_best:
                filepath = str(save_dir / 'best_model.pth')
            else:
                filepath = str(save_dir / f'checkpoint_epoch_{self.current_epoch}.pth')
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded: {filepath}")
