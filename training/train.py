"""
Ana egitim scripti.

Kullanim:
  python training/train.py --data-dir prepared_duck_3class
  python training/train.py --data-dir prepared_goose_2class
  python training/train.py --data-dir prepared_duck_3class --model resnet18 --epochs 40 --lr 3e-4
"""

import sys
import time
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

# Proje kok dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from config import parse_args
from dataset_utils import create_dataloaders
from model_utils import create_model, get_optimizer, get_scheduler
from evaluate import run_evaluation
from utils import (
    set_seed, get_device, setup_logging, save_config,
    EarlyStopping, MetricsTracker, plot_curves,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Bir epoch egitim yapar."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100

    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """Bir epoch validation yapar."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100

    return epoch_loss, epoch_acc


def main():
    cfg = parse_args()
    output_dir = Path(cfg["output_dir"])

    # Setup
    logger = setup_logging(output_dir)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    save_config(cfg, output_dir)

    logger.info(f"Dataset: {cfg['data_dir']}")
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Model:   {cfg['model_name']}")

    # Data
    logger.info("Veri yukleniyor...")
    loaders, class_names, class_weights = create_dataloaders(cfg)

    if "train" not in loaders or "val" not in loaders:
        logger.error("train veya val klasoru bulunamadi!")
        sys.exit(1)

    logger.info(f"Siniflar: {class_names}")
    logger.info(f"Train batches: {len(loaders['train'])} | Val batches: {len(loaders['val'])}")

    # Model
    model = create_model(cfg)
    model = model.to(device)

    # Loss
    if class_weights is not None and cfg["use_class_weights"]:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(f"Class weighted loss aktif")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    # Tracking
    tracker = MetricsTracker()
    early_stopping = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])

    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    logger.info(f"\nEgitim basliyor ({cfg['epochs']} epoch)...")
    logger.info("=" * 70)

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_one_epoch(
            model, loaders["val"], criterion, device
        )

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if cfg["scheduler"] == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Track
        tracker.update({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        epoch_time = time.time() - epoch_start

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Best model kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "config": cfg,
            }, str(output_dir / "best_model.pth"))
            logger.info(f"  >> En iyi model kaydedildi (val_acc: {val_acc:.2f}%)")

        # Periyodik checkpoint
        if epoch % cfg["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, str(output_dir / f"checkpoint_epoch_{epoch}.pth"))

        # Early stopping
        if early_stopping(val_acc):
            logger.info(f"\nEarly stopping! {cfg['patience']} epoch boyunca iyilesme yok.")
            break

    # Son model kaydet
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "config": cfg,
    }, str(output_dir / "last_model.pth"))

    total_time = time.time() - start_time
    logger.info(f"\nEgitim tamamlandi! Toplam sure: {total_time/60:.1f} dakika")
    logger.info(f"En iyi epoch: {best_epoch} (val_acc: {best_val_acc:.2f}%)")

    # Grafikleri kaydet
    plot_curves(tracker, output_dir)
    tracker.save(output_dir)
    logger.info("Loss ve accuracy grafikleri kaydedildi.")

    # Test degerlendirmesi
    if "test" in loaders:
        logger.info("\n" + "=" * 70)
        logger.info("TEST DEGERLENDIRMESI")
        logger.info("=" * 70)

        # En iyi modeli yukle
        checkpoint = torch.load(str(output_dir / "best_model.pth"), map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = run_evaluation(model, loaders["test"], device, class_names, output_dir)

        # Final ozet
        summary = {
            "dataset": cfg["data_dir"],
            "model": cfg["model_name"],
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "total_epochs_trained": epoch,
            "total_time_minutes": round(total_time / 60, 2),
            "test_metrics": test_metrics,
            "class_names": class_names,
            "config": cfg,
        }

        with open(output_dir / "final_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\nFinal ozet kaydedildi: {output_dir / 'final_summary.json'}")

    logger.info(f"\nTum ciktilar: {output_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
