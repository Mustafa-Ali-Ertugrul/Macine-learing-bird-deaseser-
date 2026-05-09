"""Import selected Roboflow folder-classification images as cattle classes.

Only explicitly mapped class folders are copied. Mixed or ambiguous folders are
ignored so the classification dataset does not inherit noisy labels.
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
    "Healthy": "Healthy",
    "Lumpy": "Lumpy_Skin_Disease",
    "Lumky Skin": "Lumpy_Skin_Disease",
    "lumpy skin": "Lumpy_Skin_Disease",
    "Dermatophilosis": "Dermatophilosis",
    "Ringworm": "Ringworm",
    "Pediculosis": "Pediculosis",
    "BRD": "Bovine_Respiratory_Disease",
    "(BRD) Bovine Disease Respiratory": "Bovine_Respiratory_Disease",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seen(metadata_path: Path) -> set[tuple[str, str]]:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Import Roboflow folder dataset into cattle_dataset.")
    parser.add_argument("--source", required=True, help="Extracted Roboflow folder dataset directory.")
    parser.add_argument("--output-dir", default="cattle_dataset")
    parser.add_argument("--limit-per-class", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = set(get_disease_classes("cattle"))
    missing = set(CLASS_MAP.values()) - allowed
    if missing:
        raise SystemExit(f"Mapped classes missing from cattle config: {sorted(missing)}")

    metadata_path = output_dir / "_roboflow_folder_import_metadata.jsonl"
    seen = load_seen(metadata_path)
    totals: dict[str, int] = {}

    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        for split_dir in source_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for label_dir in split_dir.iterdir():
                if not label_dir.is_dir() or label_dir.name not in CLASS_MAP:
                    continue

                class_name = CLASS_MAP[label_dir.name]
                class_dir = output_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                for image_path in label_dir.iterdir():
                    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if current_count(class_dir) >= args.limit_per_class:
                        break

                    source_file = str(image_path.relative_to(source_dir))
                    source = str(source_dir.relative_to(PROJECT_ROOT))
                    source_key = (source, source_file)
                    if source_key in seen:
                        continue

                    digest = digest_file(image_path)
                    suffix = ".jpg" if image_path.suffix.lower() == ".jpeg" else image_path.suffix.lower()
                    target_path = class_dir / f"roboflow_{digest[:16]}{suffix}"
                    if not target_path.exists():
                        shutil.copy2(image_path, target_path)

                    metadata = {
                        "source": source,
                        "source_file": source_file,
                        "class_name": class_name,
                        "sha256": digest,
                        "file_path": str(target_path),
                    }
                    metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    seen.add(source_key)
                    totals[class_name] = totals.get(class_name, 0) + 1

    print("Imported:")
    for class_name, count in sorted(totals.items()):
        print(f"  {class_name}: {count}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
