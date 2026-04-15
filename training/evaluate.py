"""
Model degerlendirme: test accuracy, macro F1, confusion matrix, classification report.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, class_names):
    """Modeli test/val seti uzerinde degerlendirir."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": round(accuracy_score(all_labels, all_preds) * 100, 2),
        "macro_precision": round(precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2),
        "macro_recall": round(recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2),
        "macro_f1": round(f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2),
    }

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    return metrics, report, cm, all_preds, all_labels


def plot_confusion_matrix(cm, class_names, output_path):
    """Confusion matrix gorsellestirir ve kaydeder."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14, pad=15)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    # Hucrelere sayilari yaz
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(model, test_loader, device, class_names, output_dir):
    """Tam degerlendirme calistirir ve sonuclari kaydeder."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Test degerlendirmesi basliyor...")
    metrics, report, cm, preds, labels = evaluate_model(model, test_loader, device, class_names)

    # Konsol
    logger.info(f"Test Accuracy:  {metrics['accuracy']}%")
    logger.info(f"Macro Precision: {metrics['macro_precision']}%")
    logger.info(f"Macro Recall:    {metrics['macro_recall']}%")
    logger.info(f"Macro F1:        {metrics['macro_f1']}%")
    logger.info(f"\n{report}")

    # metrics.json
    metrics_path = output / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # classification_report.txt
    report_path = output / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write(f"\n\nOverall Accuracy: {metrics['accuracy']}%\n")
        f.write(f"Macro Precision:  {metrics['macro_precision']}%\n")
        f.write(f"Macro Recall:     {metrics['macro_recall']}%\n")
        f.write(f"Macro F1:         {metrics['macro_f1']}%\n")

    # confusion_matrix.png
    cm_path = output / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    logger.info(f"Confusion matrix kaydedildi: {cm_path}")

    return metrics
