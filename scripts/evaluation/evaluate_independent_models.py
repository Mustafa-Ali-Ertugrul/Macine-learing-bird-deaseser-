#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate available trained models on independent labeled image folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import get_disease_classes  # noqa: E402
from train_model import build_model  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPECIES_TRAIN_ROOTS = {
    "chicken": [ROOT / "final_dataset_10_classes", ROOT / "final_dataset_clean", ROOT / "final_dataset_split"],
    "duck": [ROOT / "duck_dataset_10_classes", ROOT / "duck_split_dataset"],
    "goose": [ROOT / "goose_dataset_10_classes", ROOT / "goose_split_dataset"],
    "cattle": [ROOT / "cattle_dataset", ROOT / "cattle_split_dataset", ROOT / "cattle_augmented_split_dataset"],
}
SPECIES_TEST_ROOTS = {
    "chicken": [ROOT / "dataset" / "chicken", ROOT / "downloaded_disease_images"],
    "duck": [ROOT / "dataset" / "duck", ROOT / "downloaded_disease_images"],
    "goose": [ROOT / "dataset" / "goose", ROOT / "downloaded_disease_images"],
    "cattle": [ROOT / "cattle_dataset_web_candidates"],
}
LABEL_ALIASES = {
    "avian_pox": "Fowl_Pox",
    "Bovine_Respiratory_Disease": "Bovine_Tuberculosis",
    "Dermatophytosis": "Ringworm",
    "bumblefoot": "Bumblefoot",
    "duck_plague": "Duck_Plague",
    "goose_parvovirus": "Goose_Parvovirus",
}
HF_MODEL_IDS = {
    "vit_b16": "google/vit-base-patch16-224",
    "convnext_tiny": "facebook/convnext-tiny-224",
    "cvt_13": "microsoft/cvt-13",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    species: str
    architecture: str
    path: Path
    loader: str = "project"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def normalize_label(label: str) -> str:
    label = LABEL_ALIASES.get(label, label)
    return label.replace(" ", "_").replace("'", "").replace("-", "_")


def build_hashes(roots: list[Path]) -> set[str]:
    hashes = set()
    for root in roots:
        if root.exists():
            hashes.update(sha256(path) for path in iter_images(root))
    return hashes


def collect_samples(species: str, check_overlap: bool = False) -> tuple[list[dict], dict]:
    classes = set(get_disease_classes(species))
    train_hashes = build_hashes(SPECIES_TRAIN_ROOTS[species]) if check_overlap else set()
    samples = []
    skipped = {"unsupported": 0, "overlap": 0}

    for root in SPECIES_TEST_ROOTS[species]:
        if not root.exists():
            continue
        for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            label = normalize_label(label_dir.name)
            files = list(iter_images(label_dir))
            if label not in classes:
                skipped["unsupported"] += len(files)
                continue
            for image_path in files:
                digest = sha256(image_path)
                if check_overlap and digest in train_hashes:
                    skipped["overlap"] += 1
                    continue
                samples.append({"label": label, "path": image_path, "sha256": digest, "root": root})

    return samples, {**skipped, "train_hash_count": len(train_hashes)}


def simple_cnn(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )


def load_pytorch_model(spec: ModelSpec, class_names: list[str], device: torch.device) -> nn.Module:
    checkpoint = torch.load(spec.path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    if spec.loader == "simple":
        model = simple_cnn(len(class_names))
    elif spec.loader == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif spec.loader == "torchvision_convnext":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
    elif spec.loader == "timm_cvt":
        import timm
        try:
            model = timm.create_model("cvt_13", pretrained=False, num_classes=len(class_names))
        except Exception:
            model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=len(class_names))
    elif spec.loader == "project":
        if isinstance(checkpoint, dict):
            checkpoint_classes = checkpoint.get("class_names")
            if checkpoint_classes:
                class_names[:] = list(checkpoint_classes)
        model = build_model(spec.architecture, len(class_names), pretrained=False)
    else:
        raise ValueError(f"Unknown loader: {spec.loader}")

    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def load_hf_model(spec: ModelSpec, class_names: list[str], device: torch.device):
    from transformers import AutoModelForImageClassification

    model_id = os.fspath(spec.path) if spec.path.exists() else HF_MODEL_IDS[spec.architecture]
    model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=spec.path.exists())
    id2label = getattr(model.config, "id2label", None) or {}
    if id2label:
        class_names[:] = [normalize_label(id2label[i] if i in id2label else id2label[str(i)]) for i in range(len(id2label))]
    model.to(device).eval()
    return model


def discover_models() -> list[ModelSpec]:
    specs = [
        ModelSpec("chicken_hf_vit_b16", "chicken", "vit_b16", ROOT / "results" / "vit_poultry_results" / "final_model", "hf"),
        ModelSpec("chicken_hf_convnext_tiny", "chicken", "convnext_tiny", ROOT / "results" / "convnext_poultry_results" / "final_model", "hf"),
        ModelSpec("chicken_hf_cvt_13", "chicken", "cvt_13", ROOT / "results" / "cvt_poultry_results" / "final_model", "hf"),
        ModelSpec("chicken_pth_vit_b16", "chicken", "vit_b16", ROOT / "poultry_disease_vit.pth", "project"),
        ModelSpec("chicken_pth_resnext50", "chicken", "resnext50", ROOT / "resnext_poultry_results" / "best_resnext.pth", "project"),
        ModelSpec("chicken_pth_resnest50d", "chicken", "resnest50d", ROOT / "resnest_poultry_results" / "best_resnest.pth", "project"),
        ModelSpec("legacy_simple_cnn", "chicken", "simple", ROOT / "scripts" / "training" / "best_poultry_disease_simple.pth", "simple"),
        ModelSpec("legacy_resnet18", "chicken", "resnet18", ROOT / "scripts" / "training" / "best_poultry_disease_resnet.pth", "resnet18"),
        ModelSpec("legacy_convnext_tiny", "chicken", "convnext_tiny", ROOT / "scripts" / "training" / "best_poultry_disease_convnext.pth", "torchvision_convnext"),
        ModelSpec("legacy_cvt_13", "chicken", "cvt_13", ROOT / "scripts" / "training" / "best_poultry_disease_cvt.pth", "timm_cvt"),
        ModelSpec("cattle_efficientnet_b0", "cattle", "efficientnet_b0", ROOT / "models" / "cattle" / "efficientnet_b0_best.pth", "project"),
        ModelSpec("cattle_mobilenet_v2", "cattle", "mobilenet_v2", ROOT / "models" / "cattle" / "mobilenet_v2_best.pth", "project"),
        ModelSpec("cattle_resnet50", "cattle", "resnet50", ROOT / "models" / "cattle" / "resnet50_best.pth", "project"),
        ModelSpec("cattle_resnext50", "cattle", "resnext50", ROOT / "models" / "cattle" / "resnext50_best.pth", "project"),
        ModelSpec("cattle_vit_b16", "cattle", "vit_b16", ROOT / "models" / "cattle" / "vit_b16_best.pth", "project"),
        ModelSpec("cattle_strict_efficientnet_b0", "cattle", "efficientnet_b0", ROOT / "cattle_strict_efficientnet_kasa" / "best_model.pth", "project"),
        ModelSpec("duck_efficientnet_b0", "duck", "efficientnet_b0", ROOT / "models" / "duck" / "efficientnet_b0_best.pth", "project"),
        ModelSpec("goose_efficientnet_b0", "goose", "efficientnet_b0", ROOT / "models" / "goose" / "efficientnet_b0_best.pth", "project"),
    ]
    return [spec for spec in specs if spec.path.exists()]


def evaluate(spec: ModelSpec, samples: list[dict], device: torch.device, out_dir: Path) -> dict:
    class_names = get_disease_classes(spec.species)
    if spec.loader == "hf":
        model = load_hf_model(spec, class_names, device)
    else:
        model = load_pytorch_model(spec, class_names, device)

    class_to_idx = {normalize_label(name): idx for idx, name in enumerate(class_names)}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    expected_ids = []
    predicted_ids = []
    for sample in samples:
        expected = normalize_label(sample["label"])
        if expected not in class_to_idx:
            continue
        try:
            image = Image.open(sample["path"]).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                logits = output.logits if hasattr(output, "logits") else output
                probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            predicted = normalize_label(class_names[pred_idx])
            confidence = float(probs[pred_idx].item())
            expected_ids.append(class_to_idx[expected])
            predicted_ids.append(pred_idx)
            rows.append({
                "model": spec.name,
                "species": spec.species,
                "expected": expected,
                "predicted": predicted,
                "correct": expected == predicted,
                "confidence": confidence,
                "path": os.fspath(sample["path"]),
                "sha256": sample["sha256"],
            })
        except Exception as exc:
            rows.append({
                "model": spec.name,
                "species": spec.species,
                "expected": expected,
                "predicted": "ERROR",
                "correct": False,
                "confidence": 0.0,
                "path": os.fspath(sample["path"]),
                "sha256": sample["sha256"],
                "error": str(exc),
            })

    safe_name = spec.name.replace("/", "_")
    csv_path = out_dir / f"{safe_name}_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["model", "species", "expected", "predicted", "correct", "confidence", "path", "sha256", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    labels_present = sorted(set(expected_ids) | set(predicted_ids))
    report = classification_report(
        expected_ids,
        predicted_ids,
        labels=labels_present,
        target_names=[normalize_label(class_names[i]) for i in labels_present],
        zero_division=0,
        output_dict=True,
    ) if expected_ids else {}
    return {
        "model": spec.name,
        "species": spec.species,
        "architecture": spec.architecture,
        "loader": spec.loader,
        "checkpoint": os.fspath(spec.path),
        "tested_images": len(expected_ids),
        "accuracy": float(accuracy_score(expected_ids, predicted_ids)) if expected_ids else 0.0,
        "f1_macro": float(f1_score(expected_ids, predicted_ids, average="macro", zero_division=0)) if expected_ids else 0.0,
        "prediction_csv": os.fspath(csv_path),
        "classification_report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/independent_model_tests")
    parser.add_argument("--species", choices=["chicken", "duck", "goose", "cattle"], default=None)
    parser.add_argument("--model", default=None, help="Substring filter for model name")
    parser.add_argument(
        "--check-overlap",
        action="store_true",
        help="Hash all training images and skip exact duplicate independent samples.",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_by_species = {}
    sample_meta = {}
    for species in SPECIES_TEST_ROOTS:
        samples, meta = collect_samples(species, check_overlap=args.check_overlap)
        samples_by_species[species] = samples
        sample_meta[species] = meta | {"available_samples": len(samples)}

    summaries = []
    started = time.time()
    for spec in discover_models():
        if args.species and spec.species != args.species:
            continue
        if args.model and args.model.lower() not in spec.name.lower():
            continue
        samples = samples_by_species.get(spec.species, [])
        if not samples:
            summaries.append({"model": spec.name, "species": spec.species, "status": "skipped_no_samples"})
            continue
        print(f"Evaluating {spec.name} on {len(samples)} {spec.species} images...", flush=True)
        try:
            summaries.append(evaluate(spec, samples, device, out_dir))
        except Exception as exc:
            summaries.append({
                "model": spec.name,
                "species": spec.species,
                "architecture": spec.architecture,
                "checkpoint": os.fspath(spec.path),
                "status": "error",
                "error": str(exc),
            })

    summary = {
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "sample_meta": sample_meta,
        "models": summaries,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
