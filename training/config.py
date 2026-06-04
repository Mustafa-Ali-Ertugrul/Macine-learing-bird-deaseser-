"""
Egitim yapilandirmasi.
Komut satirindan veya dogrudan import ile kullanilir.
"""

import argparse
from pathlib import Path


def get_default_config():
    return {
        # Veri
        "data_dir": "",          # Komut satirindan verilir
        "output_dir": "",        # Komut satirindan verilir
        "num_classes": 0,        # Otomatik tespit edilir

        # Model
        "model_name": "efficientnet_b0",
        "pretrained": True,
        "freeze_backbone": False,

        # Egitim
        "epochs": 30,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "patience": 8,           # Early stopping
        "min_delta": 0.001,
        "label_smoothing": 0.0,   # 0.0-0.1 arasi
        "unfreeze_last_n": 0,     # Son N blogu ac (0 = hepsi donuk/acik)

        # Optimizer & Scheduler
        "optimizer": "adam",      # adam, adamw, sgd
        "scheduler": "cosine",   # cosine, steplr, plateau
        "step_size": 10,         # StepLR icin
        "gamma": 0.1,            # StepLR icin
        "t_max": 30,             # CosineAnnealing icin (epochs ile ayni)

        # Gorsel
        "image_size": 224,
        "use_class_weights": True,

        # Genel
        "seed": 42,
        "num_workers": 2,
        "device": "auto",        # auto, cuda, cpu
        "save_every": 5,         # Her N epoch'ta checkpoint
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Kumes hayvani hastalik siniflandirma egitimi")

    parser.add_argument("--data-dir", type=str, required=True,
                        help="Veri seti klasoru (train/val/test alt klasorleri icermeli)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Cikti klasoru (varsayilan: outputs/<dataset_adi>)")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=[
                            "efficientnet_b0",
                            "resnet50",
                            "mobilenet_v2",
                            "mobilenet_v3",
                            "resnet18",
                            "vit_b16",
                            "vit_b_16",
                        ],
                        help="Model mimarisi")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-n", type=int, default=0,
                        help="Backbone'un son N blogunu ac (freeze ile birlikte kullan)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing (0.0-0.1)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "steplr", "plateau"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    cfg = get_default_config()
    cfg["data_dir"] = args.data_dir
    cfg["model_name"] = "vit_b16" if args.model == "vit_b_16" else args.model
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["learning_rate"] = args.lr
    cfg["patience"] = args.patience
    cfg["image_size"] = args.image_size
    cfg["use_class_weights"] = not args.no_class_weights
    cfg["freeze_backbone"] = args.freeze_backbone
    cfg["unfreeze_last_n"] = args.unfreeze_last_n
    cfg["label_smoothing"] = args.label_smoothing
    cfg["weight_decay"] = args.weight_decay
    cfg["optimizer"] = args.optimizer
    cfg["scheduler"] = args.scheduler
    cfg["device"] = args.device
    cfg["seed"] = args.seed
    cfg["num_workers"] = args.num_workers
    cfg["t_max"] = args.epochs

    # Output dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    else:
        ds_name = Path(args.data_dir).name
        cfg["output_dir"] = f"outputs/{ds_name}"

    return cfg
