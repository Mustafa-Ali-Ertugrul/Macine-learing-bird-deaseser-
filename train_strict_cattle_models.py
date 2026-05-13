#!/usr/bin/env python3
"""Train and compare cattle models on the same strict train/val/test split.

This wrapper intentionally uses subprocess.run instead of os.system so failures
keep their stderr/stdout logs. It delegates training to training/train.py, which
expects an ImageFolder layout:

    cattle_strict_split/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg

Example:
    python train_strict_cattle_models.py --data-dir cattle_strict_split
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = ["efficientnet_b0", "resnet50", "mobilenet_v2", "vit_b16"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train several cattle models on one strict split and compare results."
    )
    parser.add_argument("--data-dir", default="cattle_strict_split")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def count_images(path: Path) -> int:
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def validate_split(data_dir: Path) -> list[str]:
    missing = [split for split in ("train", "val", "test") if not (data_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Strict split eksik: {data_dir}. Eksik klasorler: {', '.join(missing)}"
        )

    split_classes: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        class_dirs = sorted(path.name for path in (data_dir / split).iterdir() if path.is_dir())
        if not class_dirs:
            raise RuntimeError(f"{data_dir / split} icinde sinif klasoru yok.")
        empty_classes = [
            class_name
            for class_name in class_dirs
            if count_images(data_dir / split / class_name) == 0
        ]
        if empty_classes:
            raise RuntimeError(
                f"{data_dir / split} icinde bos siniflar var: {', '.join(empty_classes)}"
            )
        split_classes[split] = class_dirs

    if split_classes["train"] != split_classes["val"] or split_classes["train"] != split_classes["test"]:
        raise RuntimeError(
            "Strict split siniflari ayni degil.\n"
            f"train: {split_classes['train']}\n"
            f"val  : {split_classes['val']}\n"
            f"test : {split_classes['test']}"
        )
    return split_classes["train"]


def build_command(args: argparse.Namespace, model_name: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "training/train.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        os.fspath(output_dir),
        "--model",
        model_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--patience",
        str(args.patience),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.no_class_weights:
        command.append("--no-class-weights")
    return command


def load_summary(model_name: str, output_dir: Path, returncode: int) -> dict:
    summary_path = output_dir / "final_summary.json"
    row = {
        "model": model_name,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "output_dir": os.fspath(output_dir),
        "best_epoch": None,
        "best_val_acc": None,
        "test_accuracy": None,
        "test_macro_f1": None,
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("test_metrics", {})
        row.update({
            "best_epoch": summary.get("best_epoch"),
            "best_val_acc": summary.get("best_val_acc"),
            "test_accuracy": metrics.get("accuracy"),
            "test_macro_f1": metrics.get("macro_f1"),
        })
    return row


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    data_dir = Path(args.data_dir)
    class_names = validate_split(data_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Strict split: {data_dir}")
    print(f"Siniflar    : {', '.join(class_names)}")
    print(f"Modeller    : {', '.join(args.models)}")
    print("")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    rows: list[dict] = []
    for index, model_name in enumerate(args.models, start=1):
        output_dir = output_root / f"cattle_strict_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log"
        command = build_command(args, model_name, output_dir)

        print("=" * 72)
        print(f"[{index}/{len(args.models)}] {model_name}")
        print("Command:", " ".join(command))
        print("Log    :", log_path)
        print("=" * 72)

        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        rows.append(load_summary(model_name, output_dir, result.returncode))
        if result.returncode != 0:
            print(f"HATA: {model_name} basarisiz oldu. Log: {log_path}")
            if not args.continue_on_error:
                break
        else:
            print(f"TAMAM: {model_name}")

    comparison_json = output_root / "cattle_strict_model_comparison.json"
    comparison_csv = output_root / "cattle_strict_model_comparison.csv"
    comparison_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print("")
    print(f"Karsilastirma JSON: {comparison_json}")
    print(f"Karsilastirma CSV : {comparison_csv}")

    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
