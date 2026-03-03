"""Validate a 3-class YOLO dataset and report useful counts.

Expected class ids:
0 -> controlled_fire
1 -> fire
2 -> smoke
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

VALID_CLASS_IDS = {0, 1, 2}


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset labels for 3 classes")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path containing images/ and labels/ folders.",
    )
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(dataset_root / "images")
    return (dataset_root / "labels" / rel).with_suffix(".txt")


def validate(dataset_root: Path) -> int:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print("[ERROR] dataset root must contain both 'images/' and 'labels/' directories")
        return 2

    image_files = collect_images(images_dir)
    if not image_files:
        print("[ERROR] no images found under images/")
        return 2

    class_counts = Counter()
    missing_labels = 0
    malformed_lines = 0
    invalid_classes = 0

    for image in image_files:
        label_file = label_path_for_image(dataset_root, image)
        if not label_file.exists():
            missing_labels += 1
            continue

        lines = [ln.strip() for ln in label_file.read_text().splitlines() if ln.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                malformed_lines += 1
                continue
            try:
                class_id = int(parts[0])
                _ = [float(x) for x in parts[1:]]
            except ValueError:
                malformed_lines += 1
                continue

            if class_id not in VALID_CLASS_IDS:
                invalid_classes += 1
                continue

            class_counts[class_id] += 1

    print("=== Dataset validation summary ===")
    print(f"dataset root: {dataset_root}")
    print(f"images found: {len(image_files)}")
    print(f"missing label files: {missing_labels}")
    print(f"malformed label lines: {malformed_lines}")
    print(f"invalid class ids: {invalid_classes}")
    print("box counts by class:")
    print(f"  0 (controlled_fire): {class_counts[0]}")
    print(f"  1 (fire): {class_counts[1]}")
    print(f"  2 (smoke): {class_counts[2]}")

    if missing_labels or malformed_lines or invalid_classes:
        print("\n[FAIL] dataset has issues to fix before training.")
        return 1

    print("\n[PASS] dataset format looks valid for 3-class training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate(parse_args().dataset_root))
