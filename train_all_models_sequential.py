#!/usr/bin/env python3
"""Run training for one or more models sequentially.

This is the Colab-friendly entry point. It delegates the actual training to
`train_model.py`, so all model/species behavior stays in one place.

Examples:
    python train_all_models_sequential.py
    python train_all_models_sequential.py --species cattle --models vit_b16 resnet50
    python train_all_models_sequential.py --species cattle --epochs 20 --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = ["vit_b16"]
SUPPORTED_MODELS = [
    "vit_b16",
    "resnext50",
    "resnest50d",
    "convnext_tiny",
    "cvt_13",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train_model.py for selected models sequentially."
    )
    parser.add_argument(
        "--species",
        default="cattle",
        help="Species to train. Defaults to cattle.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=SUPPORTED_MODELS,
        help="Models to train sequentially.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next model if one training run fails.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, model_name: str) -> list[str]:
    command = [
        sys.executable,
        "train_model.py",
        "--model",
        model_name,
        "--species",
        args.species,
    ]

    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.no_pretrained:
        command.append("--no-pretrained")

    return command


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print("Sequential training")
    print(f"Species: {args.species}")
    print(f"Models : {', '.join(args.models)}")
    print("")

    failures: list[tuple[str, int]] = []
    for index, model_name in enumerate(args.models, start=1):
        command = build_command(args, model_name)
        print("=" * 72)
        print(f"[{index}/{len(args.models)}] Training {model_name}")
        print("Command:", " ".join(command))
        print("=" * 72)

        result = subprocess.run(command, env=env)
        if result.returncode != 0:
            failures.append((model_name, result.returncode))
            print(f"Training failed for {model_name} with exit code {result.returncode}")
            if not args.continue_on_error:
                return result.returncode

    if failures:
        print("Completed with failures:")
        for model_name, returncode in failures:
            print(f"  {model_name}: {returncode}")
        return 1

    print("All requested training runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
