#!/usr/bin/env python3
"""Audit duck/goose datasets and model checkpoints in the workspace."""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHECKPOINT_EXTS = {".pth", ".pt"}
TARGET_DIRS = [
    "dataset/duck",
    "dataset/goose",
    "cleaned_dataset/duck",
    "cleaned_dataset/goose",
    "duck_dataset_10_classes",
    "goose_dataset_10_classes",
    "duck_split_dataset",
    "goose_split_dataset",
    "prepared_duck_3class",
    "prepared_duck_3class_cleaned",
    "prepared_goose_2class",
    "prepared_goose_2class_cleaned",
    "downloaded_disease_images",
    "review",
]


def count_images_by_leaf(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
            parent = file_path.parent.name
            counts[parent] = counts.get(parent, 0) + 1
    return dict(sorted(counts.items()))


def audit_datasets() -> dict[str, dict[str, int]]:
    return {
        target: count_images_by_leaf(ROOT / target)
        for target in TARGET_DIRS
        if (ROOT / target).exists()
    }


def inspect_checkpoint(path: Path) -> dict | None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}

    info = {"path": str(path)}
    if isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names")
        config = checkpoint.get("config")
        if class_names:
            info["class_count"] = len(class_names)
            info["class_names"] = class_names
        if isinstance(config, dict):
            cfg_classes = config.get("class_names")
            if cfg_classes:
                info["config_class_count"] = len(cfg_classes)
                info["config_class_names"] = cfg_classes
            for key in ("model_name", "species", "data_dir", "output_dir"):
                if key in config:
                    info[f"config_{key}"] = config[key]
        if "model_state_dict" in checkpoint:
            info["has_model_state_dict"] = True
        if "state_dict" in checkpoint:
            info["has_state_dict"] = True
    else:
        info["type"] = type(checkpoint).__name__
    return info


def audit_checkpoints() -> list[dict]:
    rows: list[dict] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_EXTS:
            continue
        lowered = str(path).lower()
        if any(token in lowered for token in ["duck", "goose", "poultry", "models"]):
            info = inspect_checkpoint(path)
            if info:
                rows.append(info)
    return rows


def main() -> None:
    report = {
        "datasets": audit_datasets(),
        "checkpoints": audit_checkpoints(),
    }
    output = ROOT / "reports" / "duck_goose_asset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
