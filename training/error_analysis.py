"""
Egitilmis modeller icin yanlis siniflandirma analizi.

Kullanim:
  python training/error_analysis.py --model outputs/duck_3class/best_model.pth --data-dir prepared_duck_3class --output error_analysis/duck
  python training/error_analysis.py --model outputs/goose_2class/best_model.pth --data-dir prepared_goose_2class --output error_analysis/goose
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image, ImageFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import create_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_model(model_path, device):
    """Kaydedilmis modeli yukler."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", [])

    if not cfg.get("num_classes"):
        cfg["num_classes"] = len(class_names)
    if not cfg.get("model_name"):
        cfg["model_name"] = "efficientnet_b0"
    cfg["pretrained"] = False

    model = create_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, cfg


def analyze_split(model, data_dir, split, class_names, device, image_size=224):
    """Bir split'teki tum gorselleri analiz eder."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        print(f"  [UYARI] {split_dir} bulunamadi, atlaniyor.")
        return []

    transform = get_eval_transform(image_size)
    results = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)

                    # Tum sinif olasiliklari
                    all_probs = probs[0].cpu().numpy()

                results.append({
                    "filename": img_path.name,
                    "filepath": str(img_path),
                    "split": split,
                    "true_label": class_name,
                    "true_idx": class_idx,
                    "predicted_label": class_names[pred_idx.item()],
                    "predicted_idx": pred_idx.item(),
                    "confidence": round(confidence.item() * 100, 2),
                    "correct": class_idx == pred_idx.item(),
                    "class_probabilities": {
                        cn: round(float(all_probs[i]) * 100, 2)
                        for i, cn in enumerate(class_names)
                    },
                })

            except Exception as e:
                print(f"  [HATA] {img_path.name}: {e}")

    return results


def generate_reports(all_results, output_dir, class_names):
    """Tum raporlari olusturur."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    correct = [r for r in all_results if r["correct"]]
    errors = [r for r in all_results if not r["correct"]]

    # ── 1. Yanlis siniflananlari kopyala ──
    errors_dir = output / "misclassified_images"
    errors_dir.mkdir(exist_ok=True)

    for err in errors:
        src = Path(err["filepath"])
        if src.exists():
            new_name = (
                f"true_{err['true_label']}__"
                f"pred_{err['predicted_label']}__"
                f"score_{err['confidence']:.0f}__"
                f"{err['filename']}"
            )
            shutil.copy2(src, errors_dir / new_name)

    # ── 2. CSV rapor ──
    csv_path = output / "error_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "split", "true_label", "predicted_label",
            "confidence", "correct",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "filename": r["filename"],
                "split": r["split"],
                "true_label": r["true_label"],
                "predicted_label": r["predicted_label"],
                "confidence": r["confidence"],
                "correct": r["correct"],
            })

    # ── 3. Karisiklik cifti analizi ──
    confusion_pairs = defaultdict(int)
    class_errors = defaultdict(int)
    class_totals = defaultdict(int)

    for r in all_results:
        class_totals[r["true_label"]] += 1
        if not r["correct"]:
            pair = f"{r['true_label']} -> {r['predicted_label']}"
            confusion_pairs[pair] += 1
            class_errors[r["true_label"]] += 1

    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])

    # ── 4. En dusuk/yuksek confidence yanlislar ──
    errors_by_conf = sorted(errors, key=lambda x: x["confidence"])
    low_conf_errors = errors_by_conf[:5] if errors_by_conf else []
    high_conf_errors = errors_by_conf[-5:][::-1] if errors_by_conf else []

    # ── 5. JSON rapor ──
    report = {
        "summary": {
            "total_samples": len(all_results),
            "correct": len(correct),
            "errors": len(errors),
            "accuracy": round(len(correct) / len(all_results) * 100, 2) if all_results else 0,
        },
        "class_errors": {
            cn: {
                "total": class_totals[cn],
                "errors": class_errors[cn],
                "error_rate": round(class_errors[cn] / class_totals[cn] * 100, 2) if class_totals[cn] > 0 else 0,
            }
            for cn in class_names
        },
        "confusion_pairs": [
            {"pair": pair, "count": count}
            for pair, count in sorted_pairs
        ],
        "low_confidence_errors": [
            {
                "filename": e["filename"],
                "true": e["true_label"],
                "predicted": e["predicted_label"],
                "confidence": e["confidence"],
                "probabilities": e["class_probabilities"],
            }
            for e in low_conf_errors
        ],
        "high_confidence_errors": [
            {
                "filename": e["filename"],
                "true": e["true_label"],
                "predicted": e["predicted_label"],
                "confidence": e["confidence"],
                "probabilities": e["class_probabilities"],
            }
            for e in high_conf_errors
        ],
        "all_errors": [
            {
                "filename": e["filename"],
                "split": e["split"],
                "true": e["true_label"],
                "predicted": e["predicted_label"],
                "confidence": e["confidence"],
                "probabilities": e["class_probabilities"],
            }
            for e in errors
        ],
    }

    json_path = output / "error_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── 6. Confidence dagilim grafigi ──
    if errors:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Dogru vs yanlis confidence
        correct_confs = [r["confidence"] for r in correct]
        error_confs = [r["confidence"] for r in errors]

        axes[0].hist(correct_confs, bins=20, alpha=0.7, color="green", label=f"Dogru ({len(correct)})")
        axes[0].hist(error_confs, bins=20, alpha=0.7, color="red", label=f"Yanlis ({len(errors)})")
        axes[0].set_xlabel("Confidence (%)")
        axes[0].set_ylabel("Sayi")
        axes[0].set_title("Confidence Dagilimi")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Sinif bazli hata orani
        classes = list(class_names)
        error_rates = [
            class_errors[cn] / class_totals[cn] * 100 if class_totals[cn] > 0 else 0
            for cn in classes
        ]
        colors = ["#e74c3c" if r > 30 else "#f39c12" if r > 15 else "#27ae60" for r in error_rates]

        axes[1].barh(classes, error_rates, color=colors)
        axes[1].set_xlabel("Hata Orani (%)")
        axes[1].set_title("Sinif Bazli Hata Orani")
        axes[1].grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(error_rates):
            axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10)

        fig.tight_layout()
        fig.savefig(str(output / "error_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 7. TXT rapor ──
    txt_path = output / "error_analysis.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("YANLIS SINIFLANDIRMA ANALIZI\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Toplam ornek:    {len(all_results)}\n")
        f.write(f"Dogru:           {len(correct)}\n")
        f.write(f"Yanlis:          {len(errors)}\n")
        f.write(f"Accuracy:        {report['summary']['accuracy']}%\n\n")

        f.write("SINIF BAZLI HATA ORANLARI\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Sinif':<25} {'Toplam':>8} {'Hata':>8} {'Oran':>8}\n")
        f.write("-" * 50 + "\n")
        for cn in class_names:
            ce = report["class_errors"][cn]
            f.write(f"{cn:<25} {ce['total']:>8} {ce['errors']:>8} {ce['error_rate']:>7.1f}%\n")

        if sorted_pairs:
            f.write(f"\nEN COK KARISAN SINIF CIFTLERI\n")
            f.write("-" * 50 + "\n")
            for pair, count in sorted_pairs:
                f.write(f"  {pair}: {count} kez\n")

        if low_conf_errors:
            f.write(f"\nEN DUSUK GUVEN ILE YANLIS TAHMINLER\n")
            f.write("-" * 50 + "\n")
            for e in low_conf_errors:
                f.write(f"  {e['filename']}\n")
                f.write(f"    Gercek: {e['true_label']} | Tahmin: {e['predicted_label']} | Guven: {e['confidence']}%\n")

        if high_conf_errors:
            f.write(f"\nEN YUKSEK GUVEN ILE YANLIS TAHMINLER (tehlikeli)\n")
            f.write("-" * 50 + "\n")
            for e in high_conf_errors:
                f.write(f"  {e['filename']}\n")
                f.write(f"    Gercek: {e['true_label']} | Tahmin: {e['predicted_label']} | Guven: {e['confidence']}%\n")

        if errors:
            f.write(f"\nTUM YANLIS TAHMINLER\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Dosya':<35} {'Gercek':<15} {'Tahmin':<15} {'Guven':>6}\n")
            f.write("-" * 60 + "\n")
            for e in sorted(errors, key=lambda x: -x["confidence"]):
                fn = e["filename"][:33] if len(e["filename"]) > 33 else e["filename"]
                f.write(f"{fn:<35} {e['true_label']:<15} {e['predicted_label']:<15} {e['confidence']:>5.1f}%\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Yanlis siniflandirma analizi")
    parser.add_argument("--model", required=True, help="Model dosyasi (.pth)")
    parser.add_argument("--data-dir", required=True, help="Dataset klasoru")
    parser.add_argument("--output", required=True, help="Cikti klasoru")
    parser.add_argument("--splits", nargs="+", default=["test", "val"],
                        help="Analiz edilecek split'ler")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Model yukle
    print(f"Model yukleniyor: {args.model}")
    model, class_names, cfg = load_model(args.model, device)
    image_size = cfg.get("image_size", 224)
    print(f"Siniflar: {class_names}")

    # Tum split'leri analiz et
    all_results = []
    for split in args.splits:
        print(f"\n{split.upper()} seti analiz ediliyor...")
        results = analyze_split(model, args.data_dir, split, class_names, device, image_size)
        all_results.extend(results)

        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        errors = total - correct
        print(f"  {split}: {total} ornek, {correct} dogru, {errors} yanlis "
              f"({correct/total*100:.1f}%)" if total > 0 else f"  {split}: bos")

    # Raporlar
    print(f"\nRaporlar olusturuluyor: {args.output}/")
    report = generate_reports(all_results, args.output, class_names)

    # Konsol ozet
    print(f"\n{'='*60}")
    print("OZET")
    print(f"{'='*60}")
    print(f"Toplam: {report['summary']['total_samples']}")
    print(f"Dogru:  {report['summary']['correct']}")
    print(f"Yanlis: {report['summary']['errors']}")
    print(f"Acc:    {report['summary']['accuracy']}%")

    print(f"\nSinif bazli hatalar:")
    for cn, info in report["class_errors"].items():
        status = "!" if info["error_rate"] > 30 else " "
        print(f"  {status} {cn}: {info['errors']}/{info['total']} ({info['error_rate']}%)")

    if report["confusion_pairs"]:
        print(f"\nEn cok karisan ciftler:")
        for p in report["confusion_pairs"][:5]:
            print(f"  {p['pair']}: {p['count']}x")

    print(f"\nCiktilar: {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
