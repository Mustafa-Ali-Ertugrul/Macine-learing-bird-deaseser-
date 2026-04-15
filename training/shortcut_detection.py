"""
Goose modeli shortcut learning / ezberleme testi.

Testler:
  A) Grad-CAM: model gorselin neresine bakiyor?
  B) Near-duplicate: ayni vaka farkli split'lerde mi?
  C) Source leakage: kaynak bazli pattern var mi?
  D) Confidence analizi: model ne kadar emin?

Kullanim (Colab):
  python training/shortcut_detection.py \
    --model outputs/goose_best_final/best_model.pth \
    --data-dir prepared_goose_2class_cleaned \
    --output outputs/shortcut_analysis
"""

import argparse
import sys
import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import create_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_eval_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    class_names = ckpt.get("class_names", [])
    cfg["num_classes"] = len(class_names)
    cfg["pretrained"] = False
    model = create_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, cfg


def get_all_images(data_dir):
    """Tum split/class/image yollarini toplar."""
    images = []
    data_path = Path(data_dir)
    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append({
                        "path": img_path,
                        "split": split,
                        "class": class_dir.name,
                    })
    return images


# ═══════════════════════════════════════════
# A) GRAD-CAM
# ═══════════════════════════════════════════
class GradCAM:
    """EfficientNet / ResNet icin Grad-CAM."""

    def __init__(self, model, model_name):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook hedef katman
        if "efficientnet" in model_name:
            target = model.features[-1]
        elif "resnet" in model_name:
            target = model.layer4
        elif "mobilenet" in model_name:
            target = model.features[-1]
        else:
            target = list(model.children())[-2]

        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), output


def create_gradcam_grid(model, model_name, images, class_names, device,
                        output_dir, max_per_class=8, image_size=224):
    """Her sinif icin Grad-CAM grid olusturur."""
    gradcam = GradCAM(model, model_name)
    transform = get_eval_transform(image_size)

    # Sinif bazli grupla (test + val oncelikli)
    by_class = defaultdict(list)
    for info in images:
        by_class[info["class"]].append(info)

    all_results = []

    for class_name in class_names:
        class_imgs = by_class.get(class_name, [])
        # Test ve val oncelikli
        class_imgs.sort(key=lambda x: {"test": 0, "val": 1, "train": 2}.get(x["split"], 3))
        selected = class_imgs[:max_per_class]

        if not selected:
            continue

        n = len(selected)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f"Grad-CAM: {class_name} ({n} gorsel)", fontsize=16, fontweight="bold")

        for idx, info in enumerate(selected):
            img_path = info["path"]

            # Orijinal gorsel
            orig = Image.open(img_path).convert("RGB")
            orig_resized = orig.resize((image_size, image_size))

            # Tahmin
            tensor = transform(orig).unsqueeze(0).to(device)
            cam, output = gradcam.generate(tensor)
            probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
            pred_idx = probs.argmax()
            pred_class = class_names[pred_idx]
            confidence = probs[pred_idx] * 100

            # CAM overlay
            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (image_size, image_size), Image.BILINEAR)) / 255.0

            heatmap = cm.jet(cam_resized)[:, :, :3]
            orig_arr = np.array(orig_resized) / 255.0
            overlay = 0.5 * orig_arr + 0.5 * heatmap

            # Ust: orijinal
            axes[0, idx].imshow(orig_resized)
            axes[0, idx].set_title(f"{info['split']}\n{img_path.name[:20]}", fontsize=9)
            axes[0, idx].axis("off")

            # Alt: Grad-CAM overlay
            color = "green" if pred_class == class_name else "red"
            axes[1, idx].imshow(overlay)
            axes[1, idx].set_title(f"Pred: {pred_class}\n{confidence:.0f}%",
                                   fontsize=10, color=color, fontweight="bold")
            axes[1, idx].axis("off")

            # Sonuc kayit
            # CAM yogunluk analizi
            center_region = cam_resized[image_size//4:3*image_size//4,
                                         image_size//4:3*image_size//4]
            edge_region_mask = np.ones_like(cam_resized, dtype=bool)
            edge_region_mask[image_size//4:3*image_size//4,
                           image_size//4:3*image_size//4] = False
            edge_region = cam_resized[edge_region_mask]

            center_activation = float(np.mean(center_region))
            edge_activation = float(np.mean(edge_region))

            focus = "MERKEZ" if center_activation > edge_activation * 1.3 else \
                    "KENAR" if edge_activation > center_activation * 1.3 else "DAGILMIS"

            all_results.append({
                "filename": img_path.name,
                "class": class_name,
                "split": info["split"],
                "predicted": pred_class,
                "confidence": round(confidence, 1),
                "correct": pred_class == class_name,
                "center_activation": round(center_activation, 3),
                "edge_activation": round(edge_activation, 3),
                "focus_region": focus,
                "probabilities": {cn: round(float(probs[i]) * 100, 1)
                                  for i, cn in enumerate(class_names)},
            })

        fig.tight_layout()
        fig.savefig(str(output_dir / f"gradcam_{class_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grad-CAM grid kaydedildi: gradcam_{class_name}.png")

    return all_results


# ═══════════════════════════════════════════
# B) NEAR-DUPLICATE TESTI
# ═══════════════════════════════════════════
def compute_phash(img_path, hash_size=16):
    """Gorsel icin perceptual hash hesaplar."""
    try:
        img = Image.open(img_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        mean = arr.mean()
        bits = (arr > mean).flatten()
        return "".join("1" if b else "0" for b in bits)
    except Exception:
        return None


def hamming_distance(h1, h2):
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def find_near_duplicates(images, threshold=30):
    """Split'ler arasi near-duplicate bulur."""
    # Hash hesapla
    for info in images:
        info["phash"] = compute_phash(info["path"])

    cross_split_dupes = []
    within_split_dupes = []

    for i in range(len(images)):
        if images[i]["phash"] is None:
            continue
        for j in range(i + 1, len(images)):
            if images[j]["phash"] is None:
                continue

            dist = hamming_distance(images[i]["phash"], images[j]["phash"])

            if dist <= threshold:
                pair = {
                    "file_a": images[i]["path"].name,
                    "split_a": images[i]["split"],
                    "class_a": images[i]["class"],
                    "file_b": images[j]["path"].name,
                    "split_b": images[j]["split"],
                    "class_b": images[j]["class"],
                    "hamming_distance": dist,
                    "similarity": round((1 - dist / len(images[i]["phash"])) * 100, 1),
                }

                if images[i]["split"] != images[j]["split"]:
                    cross_split_dupes.append(pair)
                else:
                    within_split_dupes.append(pair)

    return cross_split_dupes, within_split_dupes


# ═══════════════════════════════════════════
# C) CONFIDENCE ANALIZI
# ═══════════════════════════════════════════
def analyze_confidence(model, images, class_names, device, image_size=224):
    """Tum gorsellerin confidence dagilimini analiz eder."""
    transform = get_eval_transform(image_size)
    results = []

    for info in images:
        try:
            img = Image.open(info["path"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()

            pred_idx = probs.argmax()
            results.append({
                "filename": info["path"].name,
                "split": info["split"],
                "true_class": info["class"],
                "predicted_class": class_names[pred_idx],
                "confidence": round(float(probs[pred_idx]) * 100, 1),
                "correct": info["class"] == class_names[pred_idx],
                "margin": round(float(probs[pred_idx] - sorted(probs)[-2]) * 100, 1),
            })
        except Exception:
            pass

    return results


def plot_confidence_analysis(conf_results, class_names, output_dir):
    """Confidence dagilim grafigi."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Split bazli confidence
    for split in ["train", "val", "test"]:
        data = [r["confidence"] for r in conf_results if r["split"] == split]
        if data:
            axes[0].hist(data, bins=20, alpha=0.6, label=f"{split} ({len(data)})")
    axes[0].set_xlabel("Confidence (%)")
    axes[0].set_ylabel("Sayi")
    axes[0].set_title("Split Bazli Confidence Dagilimi")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Sinif bazli confidence (sadece dogru tahminler)
    for cn in class_names:
        data = [r["confidence"] for r in conf_results
                if r["true_class"] == cn and r["correct"]]
        if data:
            axes[1].hist(data, bins=15, alpha=0.6, label=cn)
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Sinif Bazli Confidence (Dogru Tahminler)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Margin dagilimi
    margins = [r["margin"] for r in conf_results]
    colors = ["green" if r["correct"] else "red" for r in conf_results]
    axes[2].scatter(range(len(margins)), sorted(margins, reverse=True),
                    c=["green" if m > 50 else "orange" if m > 20 else "red"
                       for m in sorted(margins, reverse=True)],
                    alpha=0.6, s=20)
    axes[2].set_xlabel("Gorsel (sirali)")
    axes[2].set_ylabel("Confidence Margin (%)")
    axes[2].set_title("Tahmin Margin Dagilimi\n(Yesil=emin, Kirmizi=dusos)")
    axes[2].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Model Confidence Analizi", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "confidence_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Shortcut learning / ezberleme testi")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    # Model yukle
    print(f"Model: {args.model}")
    model, class_names, cfg = load_model(args.model, device)
    model_name = cfg.get("model_name", "efficientnet_b0")
    image_size = cfg.get("image_size", 224)

    # Gorselleri topla
    images = get_all_images(args.data_dir)
    print(f"Toplam gorsel: {len(images)}")
    for split in ["train", "val", "test"]:
        count = sum(1 for i in images if i["split"] == split)
        print(f"  {split}: {count}")

    # ── A) GRAD-CAM ──
    print(f"\n{'='*50}")
    print("A) GRAD-CAM ANALIZI")
    print(f"{'='*50}")
    gradcam_results = create_gradcam_grid(
        model, model_name, images, class_names, device, output_dir,
        max_per_class=10, image_size=image_size,
    )

    # Grad-CAM ozet
    focus_counts = defaultdict(int)
    for r in gradcam_results:
        focus_counts[r["focus_region"]] += 1

    print(f"\n  Odak dagilimi:")
    for focus, count in sorted(focus_counts.items()):
        pct = count / len(gradcam_results) * 100
        emoji = "✓" if focus == "MERKEZ" else "⚠" if focus == "KENAR" else "~"
        print(f"    {emoji} {focus}: {count} ({pct:.0f}%)")

    # ── B) NEAR-DUPLICATE ──
    print(f"\n{'='*50}")
    print("B) NEAR-DUPLICATE TESTI")
    print(f"{'='*50}")
    cross_dupes, within_dupes = find_near_duplicates(images, threshold=25)

    print(f"  Cross-split near-duplicates: {len(cross_dupes)}")
    for d in cross_dupes[:10]:
        print(f"    {d['file_a']} ({d['split_a']}) <-> {d['file_b']} ({d['split_b']}) "
              f"| benzerlik: {d['similarity']}%")

    if cross_dupes:
        print(f"\n  ⚠ UYARI: {len(cross_dupes)} gorsel farkli split'lerde near-duplicate!")
        print(f"    Bu 'acik kitap sinavi' etkisi yaratabilir.")

    print(f"  Within-split near-duplicates: {len(within_dupes)}")

    # ── C) CONFIDENCE ──
    print(f"\n{'='*50}")
    print("C) CONFIDENCE ANALIZI")
    print(f"{'='*50}")
    conf_results = analyze_confidence(model, images, class_names, device, image_size)
    plot_confidence_analysis(conf_results, class_names, output_dir)

    # Confidence ozet
    for split in ["train", "val", "test"]:
        split_confs = [r["confidence"] for r in conf_results if r["split"] == split]
        split_margins = [r["margin"] for r in conf_results if r["split"] == split]
        if split_confs:
            print(f"  {split}: ort confidence {np.mean(split_confs):.1f}% "
                  f"(min {min(split_confs):.1f}%, max {max(split_confs):.1f}%) "
                  f"| ort margin {np.mean(split_margins):.1f}%")

    # Overfitting sinyali: train confidence >> test confidence
    train_conf = np.mean([r["confidence"] for r in conf_results if r["split"] == "train"] or [0])
    test_conf = np.mean([r["confidence"] for r in conf_results if r["split"] == "test"] or [0])
    conf_gap = train_conf - test_conf

    # ── FINAL RAPOR ──
    print(f"\n{'='*50}")
    print("SHORTCUT LEARNING RAPORU")
    print(f"{'='*50}")

    verdict = {
        "gradcam_focus": dict(focus_counts),
        "center_pct": round(focus_counts.get("MERKEZ", 0) / max(len(gradcam_results), 1) * 100, 1),
        "cross_split_duplicates": len(cross_dupes),
        "within_split_duplicates": len(within_dupes),
        "train_avg_confidence": round(train_conf, 1),
        "test_avg_confidence": round(test_conf, 1),
        "confidence_gap": round(conf_gap, 1),
        "overfitting_risk": "YUKSEK" if conf_gap > 15 or len(cross_dupes) > 3 else
                            "ORTA" if conf_gap > 5 or len(cross_dupes) > 0 else "DUSUK",
        "shortcut_risk": "YUKSEK" if focus_counts.get("KENAR", 0) > focus_counts.get("MERKEZ", 0) else
                         "ORTA" if focus_counts.get("DAGILMIS", 0) > focus_counts.get("MERKEZ", 0) else "DUSUK",
    }

    risk_color = {"YUKSEK": "🔴", "ORTA": "🟡", "DUSUK": "🟢"}

    print(f"\n  Overfitting riski:      {risk_color.get(verdict['overfitting_risk'], '')} {verdict['overfitting_risk']}")
    print(f"  Shortcut learning riski: {risk_color.get(verdict['shortcut_risk'], '')} {verdict['shortcut_risk']}")
    print(f"  Grad-CAM merkez odak:   {verdict['center_pct']}%")
    print(f"  Cross-split duplicate:   {verdict['cross_split_duplicates']}")
    print(f"  Confidence gap:          {verdict['confidence_gap']}%")

    # Kaydet
    full_report = {
        "verdict": verdict,
        "gradcam_results": gradcam_results,
        "cross_split_duplicates": cross_dupes[:20],
        "within_split_duplicates": within_dupes[:20],
        "confidence_summary": {
            split: {
                "count": len([r for r in conf_results if r["split"] == split]),
                "avg_confidence": round(np.mean([r["confidence"] for r in conf_results
                                                  if r["split"] == split] or [0]), 1),
                "avg_margin": round(np.mean([r["margin"] for r in conf_results
                                              if r["split"] == split] or [0]), 1),
            }
            for split in ["train", "val", "test"]
        },
    }

    with open(output_dir / "shortcut_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # TXT rapor
    with open(output_dir / "shortcut_report.txt", "w", encoding="utf-8") as f:
        f.write("SHORTCUT LEARNING / EZBERLEME TESTI RAPORU\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data_dir}\n\n")

        f.write(f"SONUC\n{'-'*55}\n")
        f.write(f"Overfitting riski:       {verdict['overfitting_risk']}\n")
        f.write(f"Shortcut learning riski: {verdict['shortcut_risk']}\n")
        f.write(f"Grad-CAM merkez odak:    {verdict['center_pct']}%\n")
        f.write(f"Cross-split duplicate:   {verdict['cross_split_duplicates']}\n")
        f.write(f"Confidence gap:          {verdict['confidence_gap']}%\n\n")

        f.write(f"GRAD-CAM ODAK DAGILIMI\n{'-'*55}\n")
        for focus, count in focus_counts.items():
            f.write(f"  {focus}: {count}\n")

        if cross_dupes:
            f.write(f"\nCROSS-SPLIT NEAR-DUPLICATES ({len(cross_dupes)})\n{'-'*55}\n")
            for d in cross_dupes:
                f.write(f"  {d['file_a']} ({d['split_a']}) <-> "
                        f"{d['file_b']} ({d['split_b']}) | {d['similarity']}%\n")

    print(f"\n  Ciktilar: {output_dir}/")
    print(f"    gradcam_*.png")
    print(f"    confidence_analysis.png")
    print(f"    shortcut_report.json/txt")


if __name__ == "__main__":
    main()
