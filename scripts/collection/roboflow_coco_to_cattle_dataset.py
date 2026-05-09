"""Import selected Roboflow COCO object-detection images as cattle classes.

The training pipeline is image-classification based. This script converts only
single-label COCO images into `cattle_dataset/<class>/`. Images with no selected
label or multiple selected labels are skipped to avoid noisy classification data.

Example:
    python scripts/collection/roboflow_coco_to_cattle_dataset.py \
        --source cattle_roboflow_downloads/livestock_disease_v11/extracted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_disease_classes


CLASS_MAP = {
    "Foot and Mouth disease": "Foot_and_Mouth_Disease",
    "Lumpy skin disease": "Lumpy_Skin_Disease",
    "mastitis": "Mastitis",
    "Bovine-tuberculosis": "Bovine_Tuberculosis",
    "Bovine tuberculosis": "Bovine_Tuberculosis",
    "bovine tuberculosis": "Bovine_Tuberculosis",
    "Infected": "Digital_Dermatitis",
    "Overgrown": "Hoof_Overgrowth",
    "(BRD)": "Bovine_Respiratory_Disease",
    "Respiratory": "Bovine_Respiratory_Disease",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seen_sources(metadata_path: Path) -> set[tuple[str, str]]:
    if not metadata_path.exists():
        return set()

    seen = set()
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add((item.get("source", ""), item.get("source_file", "")))
    return seen


def current_count(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def import_annotation_file(
    annotation_path: Path,
    output_dir: Path,
    metadata_file,
    seen_sources: set[tuple[str, str]],
    limit_per_class: int,
) -> dict[str, int]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}
    images = {image["id"]: image for image in data.get("images", [])}
    labels_by_image: dict[int, set[str]] = {}

    for annotation in data.get("annotations", []):
        raw_label = categories.get(annotation.get("category_id"))
        mapped_label = CLASS_MAP.get(raw_label)
        if not mapped_label:
            continue
        labels_by_image.setdefault(annotation["image_id"], set()).add(mapped_label)

    imported: dict[str, int] = {}
    split_dir = annotation_path.parent
    source_name = str(annotation_path.parent.resolve().relative_to(PROJECT_ROOT))

    for image_id, labels in labels_by_image.items():
        if len(labels) != 1:
            continue

        class_name = next(iter(labels))
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        if current_count(class_dir) >= limit_per_class:
            continue

        image_info = images.get(image_id)
        if not image_info:
            continue

        source_file = image_info["file_name"]
        source_path = split_dir / source_file
        if not source_path.exists() or source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        source_key = (source_name, source_file)
        if source_key in seen_sources:
            continue

        digest = image_digest(source_path)
        suffix = ".jpg" if source_path.suffix.lower() == ".jpeg" else source_path.suffix.lower()
        target_path = class_dir / f"roboflow_{digest[:16]}{suffix}"
        if not target_path.exists():
            shutil.copy2(source_path, target_path)

        metadata = {
            "source": source_name,
            "source_file": source_file,
            "class_name": class_name,
            "sha256": digest,
            "file_path": str(target_path),
        }
        metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        seen_sources.add(source_key)
        imported[class_name] = imported.get(class_name, 0) + 1

    return imported


def parse_args():
    parser = argparse.ArgumentParser(description="Import selected Roboflow COCO labels into cattle_dataset.")
    parser.add_argument("--source", required=True, help="Extracted Roboflow dataset directory.")
    parser.add_argument("--output-dir", default="cattle_dataset")
    parser.add_argument("--limit-per-class", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = set(get_disease_classes("cattle"))
    unknown_targets = set(CLASS_MAP.values()) - allowed
    if unknown_targets:
        raise SystemExit(f"Mapped classes missing from cattle config: {sorted(unknown_targets)}")

    metadata_path = output_dir / "_roboflow_import_metadata.jsonl"
    seen_sources = load_seen_sources(metadata_path)
    totals: dict[str, int] = {}

    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        for annotation_path in source_dir.rglob("_annotations.coco.json"):
            imported = import_annotation_file(
                annotation_path,
                output_dir,
                metadata_file,
                seen_sources,
                args.limit_per_class,
            )
            for class_name, count in imported.items():
                totals[class_name] = totals.get(class_name, 0) + count

    print("Imported:")
    for class_name, count in sorted(totals.items()):
        print(f"  {class_name}: {count}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
