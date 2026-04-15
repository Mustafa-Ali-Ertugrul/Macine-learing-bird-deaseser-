#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common Training Utilities for Poultry Disease Classification
Provides reusable training, evaluation, and visualization functions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
import copy
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch.nn.functional as F

class TrainerBase:
    """Base class for training models"""
    
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_acc = 0.0
        self.best_model_wts = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)
        
        return val_loss, val_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, output_dir: str) -> Dict:
        """Train the model"""
        os.makedirs(output_dir, exist_ok=True)
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc.item())
            self.history['val_acc'].append(val_acc.item())
            
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print("✅ Saved best model")
        
        print(f"\nBest Val Acc: {self.best_acc:.4f}")
        return self.history
    
    def evaluate(self, test_loader: DataLoader, classes: List[str], output_dir: str):
        """Evaluate on test set"""
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        print("\n📊 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=classes))
        
        plot_confusion_matrix(all_labels, all_preds, classes, output_dir)
        
        return all_preds, all_labels

class HuggingFaceTrainer:
    """Trainer for Hugging Face models"""
    
    def __init__(self, model, train_dataset, val_dataset, output_dir: str, config: dict):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.config = config
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_num_workers=config['num_workers'],
            report_to="none",
            seed=config['random_seed']
        )
        
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    @staticmethod
    def compute_metrics(eval_pred):
        """Compute accuracy"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': (predictions == labels).mean()}
    
    def train(self):
        """Train the model"""
        print("\n🚀 Starting training...")
        self.trainer.train()
        
        final_model_path = os.path.join(self.output_dir, 'final_model')
        self.trainer.save_model(final_model_path)
        print(f"✅ Model saved to: {final_model_path}")
        
        return self.trainer
    
    def evaluate(self, test_dataset, classes: List[str]):
        """Evaluate on test set"""
        print("\n📊 Evaluating on test set...")
        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        true_labels = predictions.label_ids
        
        print("\n📊 Classification Report:")
        print(classification_report(true_labels, preds, target_names=classes))
        
        plot_confusion_matrix(true_labels, preds, classes, self.output_dir)
        
        return preds, true_labels

def plot_confusion_matrix(y_true, y_pred, classes, output_dir: str, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")

def plot_training_history(history: dict, output_dir: str):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    print(f"✅ Training history saved")

def save_results(results: dict, output_dir: str, filename: str = 'results.json'):
    """Save training results to JSON"""
    save_path = os.path.join(output_dir, filename)
    
    results_clean = {}
    for key, value in results.items():
        if isinstance(value, (np.int64, np.int32)):
            results_clean[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            results_clean[key] = float(value)
        elif isinstance(value, np.ndarray):
            results_clean[key] = value.tolist()
        else:
            results_clean[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"✅ Results saved to: {save_path}")

def print_summary(model_name: str, test_acc: float, output_dir: str, classes: List[str]):
    """Print training summary"""
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Model: {model_name}")
    print(f"📁 Model saved: {output_dir}")
    print(f"🎯 Test Accuracy: {test_acc*100:.2f}%")
    print(f"📊 Number of classes: {len(classes)}")
    print("\nClasses:")
    for i, cls in enumerate(classes, 1):
        print(f"   {i}. {cls}")
    print("=" * 60)

def setup_optimizer_scheduler(model, learning_rate: float, weight_decay: float, optimizer_type='adamw'):
    """Setup optimizer and scheduler"""
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    return optimizer, scheduler

def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset"""
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts.float())
    return class_weights
