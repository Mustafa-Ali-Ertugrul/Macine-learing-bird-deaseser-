"""
Yardimci fonksiyonlar: seed, device, logging, early stopping, plotting.
"""

import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str="auto"):
    if device_str == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            print(f"Device: CUDA ({name})")
        else:
            dev = torch.device("cpu")
            print("Device: CPU")
    else:
        dev = torch.device(device_str)
        print(f"Device: {device_str}")
    return dev


def setup_logging(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    log_file = output / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def save_config(cfg, output_dir):
    path = Path(output_dir) / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


class EarlyStopping:
    """Validation metric iyilesmezse egitimi durdurur."""

    def __init__(self, patience=8, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class MetricsTracker:
    """Egitim metriklerini takip eder."""

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

    def update(self, epoch_metrics):
        for key, val in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(val)

    def save(self, output_dir):
        path = Path(output_dir) / "training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


def plot_curves(tracker, output_dir):
    """Loss ve accuracy grafiklerini kaydeder."""
    h = tracker.history
    epochs = range(1, len(h["train_loss"]) + 1)

    # Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, h["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, h["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "loss_curve.png"), dpi=150)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, h["train_acc"], "b-o", markersize=3, label="Train Accuracy")
    ax.plot(epochs, h["val_acc"], "r-o", markersize=3, label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "accuracy_curve.png"), dpi=150)
    plt.close(fig)
