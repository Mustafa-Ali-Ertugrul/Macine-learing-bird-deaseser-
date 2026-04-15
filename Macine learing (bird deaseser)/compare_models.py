#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Farklı model + tür kombinasyonlarını karşılaştır.

Kullanım:
    python compare_models.py
    python compare_models.py --species goose
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import SUPPORTED_SPECIES, SPECIES_CONFIG, get_config

MODEL_LIST = [
    "vit_b16",
    "resnext50",
    "resnest50d",
    "convnext_tiny",
    "cvt_13",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v2",
]


def main():
    parser = argparse.ArgumentParser(
        description="Model karşılaştırma tablosu"
    )
    parser.add_argument("--species", type=str, default=None,
                        choices=SUPPORTED_SPECIES,
                        help="Belirli bir tür için filtrele")
    args = parser.parse_args()

    species_list = [args.species] if args.species else SUPPORTED_SPECIES

    print(f"\n{'='*80}")
    print(f"  📊 MODEL KARŞILAŞTIRMA TABLOSU")
    print(f"{'='*80}")
    print(f"\n  {'Tür':<12} {'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} "
          f"{'F1 Weighted':>12} {'Durum':>8}")
    print(f"  {'─'*76}")

    for species in species_list:
        display = SPECIES_CONFIG[species]["display_name"]
        for model_name in MODEL_LIST:
            config = get_config(model_name, species)
            results_path = os.path.join(
                config["output_dir"], "evaluation_results.json"
            )

            if os.path.exists(results_path):
                with open(results_path) as f:
                    results = json.load(f)
                acc = results.get("accuracy", 0)
                f1m = results.get("f1_macro", 0)
                f1w = results.get("f1_weighted", 0)
                print(
                    f"  {display:<12} {model_name:<20} {acc:>9.2%} "
                    f"{f1m:>9.4f} {f1w:>11.4f} {'✅':>8}"
                )
            else:
                model_exists = os.path.exists(config["model_save_path"])
                status = "📦" if model_exists else "❌"
                print(
                    f"  {display:<12} {model_name:<20} {'—':>10} "
                    f"{'—':>10} {'—':>12} {status:>8}"
                )

    print(f"\n  📦 = Model eğitilmiş ama değerlendirilmemiş")
    print(f"  ❌ = Model henüz eğitilmemiş\n")


if __name__ == "__main__":
    main()
