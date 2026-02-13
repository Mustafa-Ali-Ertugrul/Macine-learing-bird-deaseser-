"""Metrics computation utilities."""
import numpy as np
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary containing metrics
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class': {
            class_names[i]: {
                'precision': per_class_precision[i],
                'recall': per_class_recall[i],
                'f1': per_class_f1[i]
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def format_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> str:
    """
    Format classification report as string.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Formatted report string
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
