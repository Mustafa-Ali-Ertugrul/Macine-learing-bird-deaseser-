#!/usr/bin/env python3
"""Run the cattle EfficientNet model on sampled frames from an internet video."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from train_model import build_model


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "reports" / "internet_cow_video_test" / "pexels_herd_of_cows_855974.mp4"
MODEL_PATH = ROOT / "cattle_strict_efficientnet_kasa" / "best_model.pth"
OUTPUT_DIR = ROOT / "reports" / "internet_cow_video_test"
FRAMES_DIR = OUTPUT_DIR / "sampled_frames"
FRAME_COUNT = 12


def sample_frames() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    duration = total_frames / fps if fps else None
    indices = [
        int(round(i * (total_frames - 1) / max(FRAME_COUNT - 1, 1)))
        for i in range(FRAME_COUNT)
    ]

    paths: list[Path] = []
    for sample_no, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = FRAMES_DIR / f"frame_{sample_no:02d}_{frame_index}.jpg"
        cv2.imwrite(str(frame_path), frame)
        paths.append(frame_path)

    cap.release()

    metadata = {
        "video_path": str(VIDEO_PATH),
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": duration,
        "sampled_frame_count": len(paths),
        "sampled_frames": [str(path) for path in paths],
    }
    (OUTPUT_DIR / "video_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return paths


def load_model() -> tuple[torch.nn.Module, list[str], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model("efficientnet_b0", len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, device


def predict_frames(frame_paths: list[Path]) -> list[dict]:
    model, class_names, device = load_model()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    rows = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            top_probs, top_indices = probabilities.topk(3, dim=1)

            top3 = [
                {
                    "class": class_names[top_indices[0][i].item()],
                    "confidence": round(top_probs[0][i].item(), 4),
                }
                for i in range(3)
            ]
            rows.append(
                {
                    "frame": str(frame_path),
                    "top_prediction": top3[0]["class"],
                    "confidence": top3[0]["confidence"],
                    "top3": top3,
                }
            )

    return rows


def save_results(rows: list[dict]) -> None:
    csv_path = OUTPUT_DIR / "internet_cow_video_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "top_prediction",
                "confidence",
                "second_prediction",
                "second_confidence",
                "third_prediction",
                "third_confidence",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame": row["frame"],
                    "top_prediction": row["top3"][0]["class"],
                    "confidence": row["top3"][0]["confidence"],
                    "second_prediction": row["top3"][1]["class"],
                    "second_confidence": row["top3"][1]["confidence"],
                    "third_prediction": row["top3"][2]["class"],
                    "third_confidence": row["top3"][2]["confidence"],
                }
            )

    counts = Counter(row["top_prediction"] for row in rows)
    summary = {
        "video_source": "https://www.pexels.com/video/herd-of-cows-during-daylight-855974/",
        "video_file": str(VIDEO_PATH),
        "model_file": str(MODEL_PATH),
        "model": "efficientnet_b0",
        "frames_tested": len(rows),
        "top_prediction_counts": dict(counts),
        "average_top_confidence": round(
            sum(row["confidence"] for row in rows) / len(rows), 4
        )
        if rows
        else 0,
        "predictions": rows,
        "csv_path": str(csv_path),
    }
    (OUTPUT_DIR / "internet_cow_video_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


def main() -> None:
    frame_paths = sample_frames()
    rows = predict_frames(frame_paths)
    save_results(rows)


if __name__ == "__main__":
    main()
